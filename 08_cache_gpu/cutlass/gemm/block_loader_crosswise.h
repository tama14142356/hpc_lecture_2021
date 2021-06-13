#pragma once
#include "../util/util.h"

namespace cutlass {
namespace gemm {

template <
    int ItemsPerBlockY,
    int ItemsPerBlockK,
    int ItemsPerBlockX
    >
struct block_loader<
    ItemsPerBlockY,
    ItemsPerBlockK,
    ItemsPerBlockX,
    load_algorithm::CrosswiseCopy>
{
    //-------------------------------------------------------------------------
    // Constants and types
    //-------------------------------------------------------------------------

    enum
    {
        ItemsPerVector = 16,
        ItemsPerBlock = ItemsPerBlockK * ItemsPerBlockX
    };

    typedef io_vector<
            float,
            ItemsPerBlockK,
            ItemsPerVector>
        ldg_vector_t;

    enum
    {
        /// Number of float per ldg_vector_t
        ItemsPerVectorX = 4,

        /// Total number of ldg_vector_t within each block-wide tile
        BlockLdgVectors = divide_assert<ItemsPerBlock, ItemsPerVectorX>::value,

        /// Extent of the block-wide tile in ldg_vector_t along K-axis
        BlockLdgVectorsK = divide_assert<ItemsPerBlockK, ItemsPerVectorX>::value,

        /// Extent of the block-wide tile in ldg_vector_t along L-axis
        BlockLdgVectorsL = ItemsPerBlockX,



        /// Number of ldg_vector_t within each thread-tile
        ThreadLdgVectors = divide_assert<BlockLdgVectors, ItemsPerBlockY>::value,

        /// Extent of the thread tile in ldg_vector_t along K-axis
        ThreadLdgVectorsK = __NV_STD_MAX(1, (BlockLdgVectorsK / ItemsPerBlockY)),

        /// Extent of the thread tile in ldg_vector_t along L-axis
        ThreadLdgVectorsL = divide_assert<ThreadLdgVectors, ThreadLdgVectorsK>::value,



        /// Number of ldg_vector_t within each stripmine-tile
        StripmineLdgVectors = ItemsPerBlockY,

        /// Extent of the stripmine tile in ldg_vector_t along K-axis
        StripmineLdgVectorsK = __NV_STD_MIN(BlockLdgVectorsK, StripmineLdgVectors),

        /// Extent of the stripmine tile in ldg_vector_t along L-axis
        StripmineLdgVectorsL = divide_assert<StripmineLdgVectors, StripmineLdgVectorsK>::value,



        /// Alignment in float along L needed for committing prefetch
        AlignmentDpVectorsL = 1,
    };

    /// Predicate bit vector
    typedef uint64_t predicate_mask_t;


    //-------------------------------------------------------------------------
    // Assert assumptions
    //-------------------------------------------------------------------------

    static_assert(
        (ThreadLdgVectors <= sizeof(predicate_mask_t) * 8),
        "Predicate mask type does not contain enough bits for encoding load predicates");


    //-------------------------------------------------------------------------
    // Members
    //-------------------------------------------------------------------------

    /// Input pointer to matrix in ldg_vector_t
    ldg_vector_t *d_matrix_ldgvecs;

    /// Extent of the input matrix in ldg_vector_t along the L-axis
    int matrix_ldgvecs_l;

    /// Thread block's ending ldg_vector_t coordinate (k) within the input matrix (one-past)
    int block_end_ldgvec_k;

    /// Predicate bits for guarding ldg_vector_t loads within "whole-k" block-wide tiles
    predicate_mask_t guard;

    /// Predicate bits for guarding ldg_vector_t loads within the final block-wide "residue" tile
    predicate_mask_t residue_guard;

    /// Iteration span in "whole-k" block-wide tiles
    int wholek_tiles_remaining;

    /// Distance in ldg_vector_t within pitched-linear memory between successive coordinates along the K-axis
    int matrix_ldgvec_stride_k;

    /// Distance in ldg_vector_t within pitched-linear memory between successive coordinates along the L-axis
    int matrix_ldgvec_stride_l;

    /// ldg_vector_t coordinates (l, k) of thread-tile within the block-wide tile
    int2 block_thread_ldgvec_coords;

    /// Thread-wide tile of prefetch data
    ldg_vector_t thread_tile[ThreadLdgVectorsK][ThreadLdgVectorsL];


    //-------------------------------------------------------------------------
    // Constructor API
    //-------------------------------------------------------------------------

    /// Constructor
    inline __device__
    block_loader(
        float *d_matrix_items,        ///< Input pointer to matrix in float
        int matrix_items_l,             ///< Extent of the input matrix in float along the L-axis
        int matrix_items_stride_k,      ///< Distance in float within pitched-linear memory between successive coordinates along the K-axis
        int matrix_items_stride_l,      ///< Distance in float within pitched-linear memory between successive coordinates along the L-axis
        int matrix_block_item_coords,  ///< float coordinates (l, k) of first block-wide tile within the input matrix
        int block_end_item_k)           ///< Thread block's ending coordinate (k) within the input matrix (one-past)
    :
        block_end_ldgvec_k(block_end_item_k),
        guard(0),
        residue_guard(0)
    {
        matrix_ldgvecs_l = matrix_items_l;
        matrix_ldgvec_stride_k = matrix_items_stride_k;
        matrix_ldgvec_stride_l = (matrix_items_stride_l / ItemsPerVectorX);

        // ldg_vector_t coordinates (l, k) of thread-tile within the block-wide tile
        block_thread_ldgvec_coords = make_int2(
            (threadIdx.x / BlockLdgVectorsK),                // l-coordinate
            (threadIdx.x % BlockLdgVectorsK));               // k-coordinate

        // ldg_vector_t coordinates (l, k) of first block-wide tile within the input matrix
        int2 matrix_block_ldgvec_coords = make_int2(
            matrix_block_item_coords,                     // l-coordinate
            0);    // k-coordinate

        // Iteration span in ldg_vector_t
        int span_ldgvec_k = block_end_item_k / ItemsPerVectorX;



        // ldg_vector_t coordinates (l, k) of first thread-tile tile within the input matrix
        int2 matrix_thread_ldgvec_coords = make_int2(
            block_thread_ldgvec_coords.x + matrix_block_ldgvec_coords.x,
            block_thread_ldgvec_coords.y + matrix_block_ldgvec_coords.y);

        // Iteration range in "whole-k" block-wide tiles
        wholek_tiles_remaining = span_ldgvec_k / BlockLdgVectorsK;

        // Update the input pointer to be matrix_thread_ldgvec_coords
        this->d_matrix_ldgvecs =
            reinterpret_cast<ldg_vector_t*>(d_matrix_items) +
            (matrix_thread_ldgvec_coords.y * matrix_ldgvec_stride_k) +
            (matrix_thread_ldgvec_coords.x * matrix_ldgvec_stride_l);
    }


    //-------------------------------------------------------------------------
    // Loader API
    //-------------------------------------------------------------------------

    /**
     * Request the current block-wide tile
     */
    inline __device__
    void request()
    {
        // Outer thread-tile ldg_vector_t iteration (K-axis)
        #pragma unroll
        for (int thread_ldgvec_k = 0; thread_ldgvec_k < ThreadLdgVectorsK; ++thread_ldgvec_k)
        {
            // Inner thread-tile ldg_vector_t iteration (L-axis)
            #pragma unroll
            for (int thread_ldgvec_l = 0; thread_ldgvec_l < ThreadLdgVectorsL; ++thread_ldgvec_l)
            {
                thread_tile[thread_ldgvec_k][thread_ldgvec_l].load(
                    d_matrix_ldgvecs +
                    (thread_ldgvec_k * StripmineLdgVectorsK * matrix_ldgvec_stride_k) +
                    (thread_ldgvec_l * StripmineLdgVectorsL * matrix_ldgvec_stride_l));
            }
        }
        d_matrix_ldgvecs += (matrix_ldgvec_stride_k * BlockLdgVectorsK);
    }


    /**
     * Commit the previously-requested block-wide tile to shared memory
     *
     * NB: To facilitate padding for avoiding shared memory bank conflicts, we
     * allow the row stride SmemDpVectorsL to be arbitrarily bigger than the
     * tile width ItemsPerBlockX.
     */
    template <int SmemDpVectorsL>
    inline __device__
    void commit(
        float (&scratch_tile)[ItemsPerBlockK][SmemDpVectorsL])
    {
        static_assert(SmemDpVectorsL >= ItemsPerBlockX, "Row stride must be >= tile width.");

        // Outer thread-tile ldg_vector_t iteration (K-axis)
        #pragma unroll
        for (int thread_ldgvec_k = 0; thread_ldgvec_k < ThreadLdgVectorsK; ++thread_ldgvec_k)
        {
            int block_ldgvec_k = block_thread_ldgvec_coords.y + (thread_ldgvec_k * StripmineLdgVectorsK);

            // Inner thread-tile ldg_vector_t iteration (L-axis)
            #pragma unroll
            for (int thread_ldgvec_l = 0; thread_ldgvec_l < ThreadLdgVectorsL; ++thread_ldgvec_l)
            {
                int block_ldgvec_l = block_thread_ldgvec_coords.x + (thread_ldgvec_l * StripmineLdgVectorsL);

                // Write column of float
                #pragma unroll
                for (int dpvec = 0; dpvec < ItemsPerVectorX; ++dpvec)
                {
                    scratch_tile[(block_ldgvec_k * ItemsPerVectorX) + dpvec][block_ldgvec_l] =
                        thread_tile[thread_ldgvec_k][thread_ldgvec_l].buff[dpvec];
                }
            }
        }
    }
};


} // namespace gemm
} // namespace cutlass
