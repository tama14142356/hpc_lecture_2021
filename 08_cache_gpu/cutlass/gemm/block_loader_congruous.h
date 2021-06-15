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

      typedef io_vector<float,ItemsPerThreadX,ItemsPerVector> matrix_t;

      matrix_t *d_a;
      int stride_k;
      matrix_t thread_tile[VectorsPerThreadX];

      inline __device__
	block_loader_a_t(float *d_a, int dim_m, int block_offset) {
	  stride_k = dim_m / ItemsPerVectorX;
	  int vector_l = threadIdx.x % VectorsPerBlockX;
	  int tile_k = threadIdx.x / VectorsPerBlockX;
	  int tile_l = vector_l + block_offset / ItemsPerVectorX;
	  this->d_a = reinterpret_cast<matrix_t*>(d_a) + tile_k * stride_k + tile_l;
	}

      inline __device__
	void request() {
#pragma unroll
	  for (int i = 0; i < VectorsPerThreadX; ++i) {
	    thread_tile[i].load(d_a + (i * ThreadsPerBlockK * stride_k));
	  }
	  d_a += (stride_k * ItemsPerBlockK);
	}

      inline __device__
	void commit(float (&scratch_tile)[ItemsPerBlockK][ItemsPerBlockY]) {
	  int vector_k = threadIdx.x / VectorsPerBlockX;
	  int vector_l = threadIdx.x % VectorsPerBlockX;
#pragma unroll
	  for (int i = 0; i < VectorsPerThreadX; ++i) {
	    thread_tile[i].store(&scratch_tile[vector_k + i * ThreadsPerBlockK][vector_l * ItemsPerVectorX]);
	  }
	}
    };
  } // namespace gemm
} // namespace cutlass
