#pragma once
#include "../util/util.h"

namespace cutlass {
namespace gemm {

struct block_loader_a_t {
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
            ItemsPerThreadX,
            ItemsPerVector>
        matrix_t;

    matrix_t *d_a;
    int stride_k;
    int vector_k;
    int vector_l;
    matrix_t thread_tile[VectorsPerThreadX][1];

    inline __device__
    block_loader_a_t(float *d_a, int dim_m, int block_offset) {
        stride_k = dim_m / ItemsPerVectorX,
	vector_k = threadIdx.x / VectorsPerBlockX;
	vector_l = threadIdx.x % VectorsPerBlockX;
	int tile_k = vector_k;
	int tile_l = vector_l + block_offset / ItemsPerVectorX;
        this->d_a = reinterpret_cast<matrix_t*>(d_a) + tile_k * stride_k + tile_l;
    }

    inline __device__
    void request() {
        // Outer thread-tile matrix_t iteration (K-axis)
        #pragma unroll
        for (int thread_ldgvec_k = 0; thread_ldgvec_k < VectorsPerThreadX; ++thread_ldgvec_k)
        {
            thread_tile[thread_ldgvec_k][0].load(
                d_a +
                (thread_ldgvec_k * ThreadsPerBlockK * stride_k));
        }
        d_a += (stride_k * ItemsPerBlockK);
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

        // Outer thread-tile matrix_t iteration (K-axis)
        #pragma unroll
        for (int thread_ldgvec_k = 0; thread_ldgvec_k < VectorsPerThreadX; ++thread_ldgvec_k)
        {
            int block_ldgvec_k = vector_k + thread_ldgvec_k * ThreadsPerBlockK;
            int block_ldgvec_l = vector_l;
            thread_tile[thread_ldgvec_k][0].store(
                &scratch_tile[block_ldgvec_k][block_ldgvec_l * ItemsPerVectorX]);
        }
    }
};


} // namespace gemm
} // namespace cutlass
