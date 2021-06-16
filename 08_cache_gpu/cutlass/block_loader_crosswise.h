#pragma once

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

      struct __align__(16) fvec4 { float data[4]; };

      fvec4 *d_b;
      int stride_l;
      fvec4 tile_b[VectorsPerThreadX];

      inline __device__
	void init_b(float *d_b, int dim_k, int block_offset) {
	  stride_l = dim_k / ItemsPerVectorX;
	  int vector_l = threadIdx.x / VectorsPerBlockK;
	  int tile_k = threadIdx.x % VectorsPerBlockK;
	  int tile_l = vector_l + block_offset;
	  this->d_b = reinterpret_cast<fvec4*>(d_b) + tile_l * stride_l + tile_k;
	}

      inline __device__
	void request_b() {
#pragma unroll
	  for (int i = 0; i < VectorsPerThreadX; ++i) {
	    tile_b[i] = d_b[i * ThreadsPerBlockL * stride_l];
	  }
	  d_b += VectorsPerBlockK;
	}

      template <int ItemsPerBlockX>
	inline __device__
	void commit_b(float (&scratch_tile)[ItemsPerBlockK][ItemsPerBlockX]) {
	  int vector_k = threadIdx.x % VectorsPerBlockK;
	  int vector_l = threadIdx.x / VectorsPerBlockK;
#pragma unroll
	  for (int i = 0; i < VectorsPerThreadX; ++i) {
#pragma unroll
	    for (int j = 0; j < ItemsPerVectorX; ++j) {
	      scratch_tile[vector_k * ItemsPerVectorX + j][vector_l + i * ThreadsPerBlockL] = tile_b[i].data[j];
	    }
	  }
	}
    };
  } // namespace gemm
} // namespace cutlass
