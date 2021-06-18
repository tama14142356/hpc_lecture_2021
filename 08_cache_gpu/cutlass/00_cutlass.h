#pragma once

#include <stdint.h>

namespace cutlass {
  enum
  {
    ItemsPerVector = 4,
    VectorsPerThread = 2,
    ItemsPerThread = VectorsPerThread * ItemsPerVector, // 8

    ThreadsPerWarpY = 4, // #cols
    ThreadsPerWarpX = 8, // #rows
    ThreadsPerWarp = ThreadsPerWarpX * ThreadsPerWarpY, // 32

    WarpsPerBlockY = 2, // #cols
    WarpsPerBlockX = 1, // #rows
    ThreadsPerBlock = 64,

    ItemsPerWarpY = ThreadsPerWarpY * ItemsPerThread, // 32
    ItemsPerWarpX = ThreadsPerWarpX * ItemsPerThread, // 64
    ItemsPerBlockX = WarpsPerBlockX * ItemsPerWarpX, // 64

    Ktile = 8,
    VectorsPerMtile = ThreadsPerWarpX * VectorsPerThread, // 16 A #rows
    ThreadsPerKtile = ThreadsPerBlock / VectorsPerMtile, // 4 A #cols
    VectorsPerKtile = Ktile / ItemsPerVector, // 2 B #rows
    ThreadsPerNtile = ThreadsPerBlock / VectorsPerKtile // 32 B #cols
  };

  inline __device__
    void store(float *ptr, const float &src) {
      asm volatile ("st.global.cg.f32 [%0], %1;\n"
		    : :
		    "l"(ptr),
		    "f"(src));
    }

  inline __device__
    static void gemm(const float &a,
		     const float &b,
		     float &c) {
      asm volatile ( "fma.rn.f32 %0, %1, %2, %3;\n"
		     : "=f"(c) : "f"(a), "f"(b), "f"(c));
    }

  __global__ void kernel(int dim_m,
			 int dim_n,
			 int dim_k,
			 float *d_a,
			 float *d_b,
			 float *d_c) {
    int warp_id = threadIdx.x / ThreadsPerWarp; // 2
    int warp_x = warp_id % WarpsPerBlockX; // 2
    int warp_y = warp_id / WarpsPerBlockX; // 1
    int lane_id = threadIdx.x % ThreadsPerWarp; // 32
    int lane_x = lane_id / ThreadsPerWarpY; // 8
    int lane_y = lane_id % ThreadsPerWarpY; // 4
    int offset_y = lane_y * ItemsPerVector + warp_y * ItemsPerWarpY;
    int offset_x = lane_x * ItemsPerVector + warp_x * ItemsPerWarpX;

    struct __align__(16) vec_t { float d[ItemsPerVector]; };
    vec_t *tile_a;
    vec_t *tile_b;
    vec_t __align__(16) thread_a[VectorsPerThread];
    vec_t __align__(16) thread_b[VectorsPerThread];
    __shared__ float __align__(16) block_a[Ktile][ItemsPerBlockX];
    __shared__ float __align__(16) block_b[Ktile][ItemsPerBlockX];
    float __align__(16) fragment_c[ItemsPerThread][ItemsPerThread];

    int offset_a_m = ItemsPerBlockX * blockIdx.x / ItemsPerVector;
    int offset_b_n = ItemsPerBlockX * blockIdx.y;
    int lda = dim_m / ItemsPerVector;
    int ldb = dim_k / ItemsPerVector;
    int a_m = threadIdx.x % VectorsPerMtile; // 16
    int a_k = threadIdx.x / VectorsPerMtile; // 4
    int b_k = threadIdx.x % VectorsPerKtile; // 2
    int b_n = threadIdx.x / VectorsPerKtile; // 32
    tile_a = reinterpret_cast<vec_t*>(&d_a[(a_k * lda + (a_m + offset_a_m)) * ItemsPerVector]);
    tile_b = reinterpret_cast<vec_t*>(&d_b[((b_n + offset_b_n) * ldb + b_k) * ItemsPerVector]);
    for (int y = 0; y < ItemsPerThread; ++y)
      for (int x = 0; x < ItemsPerThread; ++x)
	fragment_c[y][x] = float(0);
    int stride_a = 0;
    int stride_b = 0;
    for (int kk = 0; kk < dim_k; kk += Ktile) {
      for (int i = 0; i < VectorsPerThread; ++i) {
	thread_a[i] = tile_a[stride_a + i * ThreadsPerKtile * lda];
	thread_b[i] = tile_b[stride_b + i * ThreadsPerNtile * ldb];
      }
      stride_a += lda * Ktile;
      stride_b += Ktile / ItemsPerVector;
      __syncthreads();
      for (int i = 0; i < VectorsPerThread; ++i) {
	for (int j = 0; j < ItemsPerVector; ++j) {
	  block_a[a_k + i * ThreadsPerKtile][a_m * ItemsPerVector + j] = thread_a[i].d[j];
	  block_b[b_k * ItemsPerVector + j][b_n + i * ThreadsPerNtile] = thread_b[i].d[j];
	}
      }
      __syncthreads();
#pragma unroll
      for (int k = 0; k < Ktile; k++) {
	float __align__(16) fragment_a[ItemsPerThread];
	float __align__(16) fragment_b[ItemsPerThread];
	for (int i = 0; i < VectorsPerThread; ++i) {
	  for (int j = 0; j < ItemsPerVector; ++j) {
	    fragment_a[i * ItemsPerVector + j] = block_a[k][offset_y + (i * ThreadsPerWarpY * ItemsPerVector) + j];
	    fragment_b[i * ItemsPerVector + j] = block_b[k][offset_x + (i * ThreadsPerWarpX * ItemsPerVector) + j];
	  }
	}
	for (int m = 0; m < ItemsPerThread; ++m) {
	  for (int n = 0; n < ItemsPerThread; ++n) {
	    gemm(fragment_a[m], fragment_b[n], fragment_c[m][n]);
	  }
	}
      }
    }
    for (int ix = 0; ix < ItemsPerThread; ++ix) {
      for (int iy = 0; iy < ItemsPerThread; iy += ItemsPerVector) {
	int vx = ix / ItemsPerVector;
	int vy = iy / ItemsPerVector;
	int tx = offset_x + (vx * ThreadsPerWarpX * ItemsPerVector) + (ix % ItemsPerVector);
	int ty = offset_y + (vy * ThreadsPerWarpY * ItemsPerVector) + (iy % ItemsPerVector);
	int bx = ItemsPerBlockX * blockIdx.y + tx;
	int by = ItemsPerBlockX * blockIdx.x + ty;
	for (int i = 0; i < ItemsPerVector; ++i) {
	  if (bx < dim_n && (by + i) < dim_m) {
	    store(d_c + bx * dim_m + by + i, fragment_c[iy + i][ix]);
	  }
	}
      }
    }
  }
} // namespace cutlass
