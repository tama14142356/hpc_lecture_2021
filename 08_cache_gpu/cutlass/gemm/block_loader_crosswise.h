#pragma once
#include "../util/util.h"

namespace cutlass {
  namespace gemm {

    struct block_loader_b_t {
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

      typedef io_vector<float,ItemsPerBlockK,ItemsPerVector> matrix_t;

      matrix_t *d_b;
      int stride_l;
      int vector_k;
      int vector_l;
      matrix_t thread_tile[VectorsPerThreadX];

      inline __device__
	block_loader_b_t(float *d_b, int dim_k, int block_offset) {
	  stride_l = dim_k / ItemsPerVectorX;
	  vector_k = threadIdx.x % VectorsPerBlockK;
	  vector_l = threadIdx.x / VectorsPerBlockK;
	  int tile_k = vector_k;
	  int tile_l = vector_l + block_offset;
	  this->d_b = reinterpret_cast<matrix_t*>(d_b) + tile_l * stride_l + tile_k;
	}

      inline __device__
	void request() {
#pragma unroll
	  for (int i = 0; i < VectorsPerThreadX; ++i) {
	    thread_tile[i].load(d_b + (i * ThreadsPerBlockL * stride_l));
	  }
	  d_b += VectorsPerBlockK;
	}

      template <int SmemDpVectorsL>
	inline __device__
	void commit(float (&scratch_tile)[ItemsPerBlockK][SmemDpVectorsL]) {
	  int block_ldgvec_k = vector_k;
#pragma unroll
	  for (int i = 0; i < VectorsPerThreadX; ++i) {
	    int block_ldgvec_l = vector_l + (i * ThreadsPerBlockL);
#pragma unroll
	    for (int j = 0; j < ItemsPerVectorX; ++j) {
	      scratch_tile[(block_ldgvec_k * ItemsPerVectorX) + j][block_ldgvec_l] =
		thread_tile[i].buff[j];
	    }
	  }
	}
    };
  } // namespace gemm
} // namespace cutlass
