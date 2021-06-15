#pragma once
#include "../util/util.h"

namespace cutlass {
namespace gemm {

template<>
struct block_loader<load_algorithm::CrosswiseCopy> {
    enum
    {
        ItemsPerVectorY = 4,
        ItemsPerVectorX = 4,
        ItemsPerVector = ItemsPerVectorX * ItemsPerVectorY, // 16
        VectorsPerThreadY = 2,
        VectorsPerThreadX = 2,
        ThreadsPerWarpY = 4,
        ThreadsPerWarpX = 8,
        ThreadsPerWarp = ThreadsPerWarpX * ThreadsPerWarpY, // 32
        WarpsPerBlockY = 2,
        WarpsPerBlockX = 1,
        ItemsPerThreadY = VectorsPerThreadY * ItemsPerVectorY, // 8
        ItemsPerThreadX = VectorsPerThreadX * ItemsPerVectorX, // 8
        ItemsPerWarpY = ThreadsPerWarpY * ItemsPerThreadY, // 32
        ItemsPerWarpX = ThreadsPerWarpX * ItemsPerThreadX, // 64
        ItemsPerBlockY = WarpsPerBlockY * ItemsPerWarpY, // 64
        ItemsPerBlockX = WarpsPerBlockX * ItemsPerWarpX, // 64
        ItemsPerBlockK = 8,
	ThreadsPerBlock = ThreadsPerWarp * WarpsPerBlockY * WarpsPerBlockX, // 64
        ItemsPerBlock = ItemsPerBlockK * ItemsPerBlockX, // 512
	VectorsPerBlockX = ItemsPerBlockX / ItemsPerVectorX, // 16
        VectorsPerBlockK = ItemsPerBlockK / ItemsPerVectorX, // 2
	VectorsPerBlock = ItemsPerBlock / ItemsPerVectorX, // 128
        ThreadsPerBlockK = ThreadsPerBlock / VectorsPerBlockX, // 4
        ThreadsPerBlockL = ThreadsPerBlock / VectorsPerBlockK // 32
    };

    typedef io_vector<
            float,
            ItemsPerBlockK,
            ItemsPerVector>
        ldg_vector_t;

    /// Input pointer to matrix in ldg_vector_t
    ldg_vector_t *d_matrix_ldgvecs;

    /// Distance in ldg_vector_t within pitched-linear memory between successive coordinates along the K-axis
    int stride_k;

    /// Distance in ldg_vector_t within pitched-linear memory between successive coordinates along the L-axis
    int stride_l;

    /// ldg_vector_t coordinates (l, k) of thread-tile within the block-wide tile
    int2 block_thread_ldgvec_coords;

    /// Thread-wide tile of prefetch data
    ldg_vector_t thread_tile[1][VectorsPerThreadX];


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
    {
        stride_k = matrix_items_stride_k;
        stride_l = (matrix_items_stride_l / ItemsPerVectorX);

        // ldg_vector_t coordinates (l, k) of thread-tile within the block-wide tile
        block_thread_ldgvec_coords = make_int2(
            (threadIdx.x / VectorsPerBlockK),                // l-coordinate
            (threadIdx.x % VectorsPerBlockK));               // k-coordinate

        // ldg_vector_t coordinates (l, k) of first block-wide tile within the input matrix
        int2 matrix_block_ldgvec_coords = make_int2(
            matrix_block_item_coords,                     // l-coordinate
            0);    // k-coordinate

        // ldg_vector_t coordinates (l, k) of first thread-tile tile within the input matrix
        int2 matrix_thread_ldgvec_coords = make_int2(
            block_thread_ldgvec_coords.x + matrix_block_ldgvec_coords.x,
            block_thread_ldgvec_coords.y + matrix_block_ldgvec_coords.y);

        // Update the input pointer to be matrix_thread_ldgvec_coords
        this->d_matrix_ldgvecs =
            reinterpret_cast<ldg_vector_t*>(d_matrix_items) +
            (matrix_thread_ldgvec_coords.y * stride_k) +
            (matrix_thread_ldgvec_coords.x * stride_l);
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
        // Inner thread-tile ldg_vector_t iteration (L-axis)
        #pragma unroll
        for (int thread_ldgvec_l = 0; thread_ldgvec_l < VectorsPerThreadX; ++thread_ldgvec_l)
        {
            thread_tile[0][thread_ldgvec_l].load(
                d_matrix_ldgvecs +
                (thread_ldgvec_l * ThreadsPerBlockL * stride_l));
        }
        d_matrix_ldgvecs += (stride_k * VectorsPerBlockK);
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
        int block_ldgvec_k = block_thread_ldgvec_coords.y;
        #pragma unroll
        for (int thread_ldgvec_l = 0; thread_ldgvec_l < VectorsPerThreadX; ++thread_ldgvec_l)
        {
            int block_ldgvec_l = block_thread_ldgvec_coords.x + (thread_ldgvec_l * ThreadsPerBlockL);
            #pragma unroll
            for (int dpvec = 0; dpvec < ItemsPerVectorX; ++dpvec)
            {
                scratch_tile[(block_ldgvec_k * ItemsPerVectorX) + dpvec][block_ldgvec_l] =
                    thread_tile[0][thread_ldgvec_l].buff[dpvec];
            }
        }
    }
};


} // namespace gemm
} // namespace cutlass
