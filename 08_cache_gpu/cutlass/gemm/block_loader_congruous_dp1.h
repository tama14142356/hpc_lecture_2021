#pragma once
#include "../util/util.h"

namespace cutlass {
namespace gemm {

template<>
struct block_loader<
    load_algorithm::CongruousCopy>  ///< Algorithm for loading a shared tile of KxL matrix data (CongruousCopy specialization)
{
    enum
    {
        ItemsPerVectorY = 4,
        ItemsPerVectorX = 4,
        VectorsPerThreadY = 2,
        VectorsPerThreadX = 2,
        ThreadsPerWarp = 32,
        ThreadsPerWarpY = 4,
        ThreadsPerWarpX = ThreadsPerWarp / ThreadsPerWarpY, // 8
        WarpsPerBlockY = 2,
        WarpsPerBlockX = 1,
        ItemsPerThreadY = VectorsPerThreadY * ItemsPerVectorY, // 8
        ItemsPerThreadX = VectorsPerThreadX * ItemsPerVectorX, // 8
        ItemsPerWarpY = ThreadsPerWarpY * ItemsPerThreadY, // 32
        ItemsPerWarpX = ThreadsPerWarpX * ItemsPerThreadX, // 64
        ItemsPerBlockY = WarpsPerBlockY * ItemsPerWarpY, // 64
        ItemsPerBlockX = WarpsPerBlockX * ItemsPerWarpX, // 64
	ThreadsPerBlock = ThreadsPerWarp * WarpsPerBlockY * WarpsPerBlockX, // 64
        ItemsPerBlockK = 8,

        ItemsPerVector = ItemsPerVectorX * ItemsPerVectorY, // 16
        ItemsPerBlock = ItemsPerBlockK * ItemsPerBlockX, // 512
        ItemsPerThread = ItemsPerBlock / ThreadsPerBlock, // 8
	VectorsPerBlock = ItemsPerBlock / ItemsPerVectorX, // 128
	VectorsPerBlockX = ItemsPerBlockX / ItemsPerVectorX // 16
    };

    typedef io_vector<
            float,
            __NV_STD_MIN(ItemsPerThread, ItemsPerBlockX),
            ItemsPerVector>
        ldg_vector_t;

    enum
    {
        VectorsPerThread = VectorsPerBlock / ThreadsPerBlock,
        VectorsPerThreadK = VectorsPerThread,

        /// Extent of the stripmine tile in ldg_vector_t along L-axis
        ThreadsPerBlockL = VectorsPerBlockX, //ThreadsPerBlock),

        /// Extent of the stripmine tile in ldg_vector_t along K-axis
        ThreadsPerBlockK = divide_assert<ThreadsPerBlock, ThreadsPerBlockL>::value,



        /// Alignment in float along L needed for committing prefetch
        AlignmentDpVectorsL = ItemsPerVectorX,
    };

    /// Predicate bit vector
    typedef uint64_t predicate_mask_t;


    //-------------------------------------------------------------------------
    // Assert assumptions
    //-------------------------------------------------------------------------

    static_assert(
        (VectorsPerThread <= sizeof(predicate_mask_t) * 8),
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
    ldg_vector_t thread_tile[VectorsPerThreadK][1];


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
        matrix_ldgvecs_l = matrix_items_l / ItemsPerVectorX;
        matrix_ldgvec_stride_k = matrix_items_stride_k / ItemsPerVectorX,
        matrix_ldgvec_stride_l = matrix_items_stride_l;

        // ldg_vector_t coordinates (l, k) of thread-tile within the block-wide tile
        block_thread_ldgvec_coords = make_int2(
            threadIdx.x % VectorsPerBlockX,                 // l-coordinate
            threadIdx.x / VectorsPerBlockX);                // k-coordinate

        // ldg_vector_t coordinates (l, k) of first block-wide tile within the input matrix
        int2 matrix_block_ldgvec_coords = make_int2(
            matrix_block_item_coords / ItemsPerVectorX,     // l-coordinate
            0);                    // k-coordinate

        // Iteration span in ldg_vector_t
        int span_ldgvec_k = block_end_item_k;



        // ldg_vector_t coordinates (l, k) of first thread-tile tile within the input matrix
        int2 matrix_thread_ldgvec_coords = make_int2(
            block_thread_ldgvec_coords.x + matrix_block_ldgvec_coords.x,
            block_thread_ldgvec_coords.y + matrix_block_ldgvec_coords.y);

        // Iteration range in "whole-k" block-wide tiles
        wholek_tiles_remaining = span_ldgvec_k / ItemsPerBlockK;

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
        for (int thread_ldgvec_k = 0; thread_ldgvec_k < VectorsPerThreadK; ++thread_ldgvec_k)
        {
            thread_tile[thread_ldgvec_k][0].load(
                d_matrix_ldgvecs +
                (thread_ldgvec_k * ThreadsPerBlockK * matrix_ldgvec_stride_k));
        }
        d_matrix_ldgvecs += (matrix_ldgvec_stride_k * ItemsPerBlockK);
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
        for (int thread_ldgvec_k = 0; thread_ldgvec_k < VectorsPerThreadK; ++thread_ldgvec_k)
        {
            int block_ldgvec_k = block_thread_ldgvec_coords.y + (thread_ldgvec_k * ThreadsPerBlockK);
            int block_ldgvec_l = block_thread_ldgvec_coords.x;
            thread_tile[thread_ldgvec_k][0].store(
                &scratch_tile[block_ldgvec_k][block_ldgvec_l * ItemsPerVectorX]);
        }
    }
};


} // namespace gemm
} // namespace cutlass
