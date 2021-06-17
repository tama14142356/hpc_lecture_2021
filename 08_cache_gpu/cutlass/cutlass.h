#pragma once

#include <stdint.h>

namespace cutlass {
  enum
  {
    ItemsPerVectorX = 4,
    ItemsPerVector = ItemsPerVectorX * ItemsPerVectorX, // 16
    VectorsPerThreadX = 2,
    ThreadsPerWarpY = 4,
    ThreadsPerWarpX = 8,
    ThreadsPerWarp = ThreadsPerWarpX * ThreadsPerWarpY, // 32
    WarpsPerBlockY = 2,
    WarpsPerBlockX = 1,
    ItemsPerThreadX = VectorsPerThreadX * ItemsPerVectorX, // 8
    ItemsPerWarpY = ThreadsPerWarpY * ItemsPerThreadX, // 32
    ItemsPerWarpX = ThreadsPerWarpX * ItemsPerThreadX, // 64
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
    int warp_id = threadIdx.x / ThreadsPerWarp;
    int warp_x = warp_id % WarpsPerBlockX;
    int warp_y = warp_id / WarpsPerBlockX;
    int lane_id = threadIdx.x % ThreadsPerWarp;
    int lane_x = lane_id / ThreadsPerWarpY;
    int lane_y = lane_id % ThreadsPerWarpY;
    int offset_y = lane_y * ItemsPerVectorX + warp_y * ItemsPerWarpY;
    int offset_x = lane_x * ItemsPerVectorX + warp_x * ItemsPerWarpX;

    float *global_a;
    float *global_b;
    float __align__(16) thread_a[VectorsPerThreadX][ItemsPerVectorX];
    float __align__(16) thread_b[VectorsPerThreadX][ItemsPerVectorX];
    __shared__ float __align__(16) block_a[ItemsPerBlockK][ItemsPerBlockX];
    __shared__ float __align__(16) block_b[ItemsPerBlockK][ItemsPerBlockX];
    float __align__(16) slice_a[2][VectorsPerThreadX][ItemsPerVectorX];
    float __align__(16) slice_b[2][VectorsPerThreadX][ItemsPerVectorX];
    float __align__(16) tile_c[ItemsPerThreadX][ItemsPerThreadX];

    int offset_a = ItemsPerBlockX * blockIdx.x;
    int offset_b = ItemsPerBlockX * blockIdx.y;
    int stride_k = dim_m / ItemsPerVectorX;
    int stride_l = dim_k / ItemsPerVectorX;
    int vector_a = threadIdx.x % VectorsPerBlockX;
    int vector_b = threadIdx.x / VectorsPerBlockK;
    int a_k = threadIdx.x / VectorsPerBlockX;
    int b_k = threadIdx.x % VectorsPerBlockK;
    int a_l = threadIdx.x % VectorsPerBlockX;
    int b_l = threadIdx.x / VectorsPerBlockK;
    int a_m = vector_a + offset_a / ItemsPerVectorX;
    int b_m = vector_b + offset_b;
    global_a = &d_a[(a_k * stride_k + a_m) * ItemsPerVectorX];
    global_b = &d_b[(b_m * stride_l + b_k) * ItemsPerVectorX];
#pragma unroll
    for (int i = 0; i < VectorsPerThreadX; ++i) {
      for (int j = 0; j < ItemsPerVectorX; ++j) {
        thread_a[i][j] = global_a[(i * ThreadsPerBlockK * stride_k) * ItemsPerVectorX + j];
        thread_b[i][j] = global_b[(i * ThreadsPerBlockL * stride_l) * ItemsPerVectorX + j];
      }
    }
    int stride_a = (stride_k * ItemsPerBlockK);
    int stride_b = VectorsPerBlockK;
#pragma unroll
    for (int i = 0; i < VectorsPerThreadX; ++i) {
#pragma unroll
      for (int j = 0; j < ItemsPerVectorX; ++j) {
	block_a[a_k + i * ThreadsPerBlockK][a_l * ItemsPerVectorX + j] = thread_a[i][j];
	block_b[b_k * ItemsPerVectorX + j][b_l + i * ThreadsPerBlockL] = thread_b[i][j];
      }
    }
    __syncthreads();
#pragma unroll
    for (int y = 0; y < ItemsPerThreadX; ++y)
#pragma unroll
      for (int x = 0; x < ItemsPerThreadX; ++x)
	tile_c[y][x] = float(0);
    for (int i = 0; i < VectorsPerThreadX; ++i) {
      for (int j = 0; j < ItemsPerVectorX; ++j) {
        slice_a[0][i][j] = block_a[0][offset_y + (i * ThreadsPerWarpY * ItemsPerVectorX) + j];
        slice_b[0][i][j] = block_b[0][offset_x + (i * ThreadsPerWarpX * ItemsPerVectorX) + j];
      }
    }
#pragma unroll
    for (int kk = 0; kk < dim_k; kk += ItemsPerBlockK) {
#pragma unroll
      for (int k = 0; k < ItemsPerBlockK; k++) {
	if ((k == ItemsPerBlockK - 1) && kk < dim_k-ItemsPerBlockK) {
	  __syncthreads();
#pragma unroll
	  for (int i = 0; i < VectorsPerThreadX; ++i) {
#pragma unroll
	    for (int j = 0; j < ItemsPerVectorX; ++j) {
	      block_a[a_k + i * ThreadsPerBlockK][a_l * ItemsPerVectorX + j] = thread_a[i][j];
	      block_b[b_k * ItemsPerVectorX + j][b_l + i * ThreadsPerBlockL] = thread_b[i][j];
	    }
	  }
	  __syncthreads();
	}
	if ((k == 0) && kk < dim_k-ItemsPerBlockK) {
#pragma unroll
	  for (int i = 0; i < VectorsPerThreadX; ++i) {
	    for (int j = 0; j < ItemsPerVectorX; ++j) {
	      thread_a[i][j] = global_a[(stride_a + i * ThreadsPerBlockK * stride_k) * ItemsPerVectorX + j];
	      thread_b[i][j] = global_b[(stride_b + i * ThreadsPerBlockL * stride_l) * ItemsPerVectorX + j];
	    }
	  }
	  stride_a += (stride_k * ItemsPerBlockK);
	  stride_b += VectorsPerBlockK;
	}
	int k1 = (k + 1) % ItemsPerBlockK;
	for (int i = 0; i < VectorsPerThreadX; ++i) {
	  for (int j = 0; j < ItemsPerVectorX; ++j) {
	    slice_a[(k + 1) % 2][i][j] = block_a[k1][offset_y + (i * ThreadsPerWarpY * ItemsPerVectorX) + j];
	    slice_b[(k + 1) % 2][i][j] = block_b[k1][offset_x + (i * ThreadsPerWarpX * ItemsPerVectorX) + j];
	  }
	}
	typedef float __align__(16) tile_t[ItemsPerThreadX];
	tile_t &tile_a = reinterpret_cast<tile_t&>(slice_a[k % 2]);
	tile_t &tile_b = reinterpret_cast<tile_t&>(slice_b[k % 2]);
#pragma unroll
	for (int y = 0; y < ItemsPerThreadX; ++y) {
#pragma unroll
	  for (int x = 0; x < ItemsPerThreadX; ++x) {
	    gemm(tile_a[y], tile_b[x], tile_c[y][x]);
	  }
	}
      }
    }
#pragma unroll
    for (int ix = 0; ix < ItemsPerThreadX; ++ix) {
#pragma unroll
      for (int iy = 0; iy < ItemsPerThreadX; iy += ItemsPerVectorX) {
	int vx = ix / ItemsPerVectorX;
	int vy = iy / ItemsPerVectorX;
	int tx = offset_x + (vx * ThreadsPerWarpX * ItemsPerVectorX) + (ix % ItemsPerVectorX);
	int ty = offset_y + (vy * ThreadsPerWarpY * ItemsPerVectorX) + (iy % ItemsPerVectorX);
	int bx = ItemsPerBlockX * blockIdx.y + tx;
	int by = ItemsPerBlockX * blockIdx.x + ty;
#pragma unroll
	for (int i = 0; i < ItemsPerVectorX; ++i) {
	  if (bx < dim_n && (by + i) < dim_m) {
	    store(d_c + bx * dim_m + by + i, tile_c[iy + i][ix]);
	  }
	}
      }
    }
  }
} // namespace cutlass
