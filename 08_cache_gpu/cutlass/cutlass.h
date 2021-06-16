#pragma once

#include <stdint.h>

namespace cutlass {
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

  typedef float __align__(16) block_y_t[ItemsPerBlockK][ItemsPerBlockY];
  typedef float __align__(16) block_x_t[ItemsPerBlockK][ItemsPerBlockX];

  typedef float tile_a_t[ItemsPerThreadY];
  typedef float tile_b_t[ItemsPerThreadX];

  inline __device__
    void init(float *d_a, int dim_m, int offset_a, int &stride_k, fvec4 **global_a,
              float *d_b, int dim_k, int offset_b, int &stride_l, fvec4 **global_b) {
      stride_k = dim_m / ItemsPerVectorX;
      stride_l = dim_k / ItemsPerVectorX;
      int vector_a = threadIdx.x % VectorsPerBlockX;
      int vector_b = threadIdx.x / VectorsPerBlockK;
      int a_k = threadIdx.x / VectorsPerBlockX;
      int b_k = threadIdx.x % VectorsPerBlockK;
      int a_l = vector_a + offset_a / ItemsPerVectorX;
      int b_l = vector_b + offset_b;
      *global_a = reinterpret_cast<fvec4*>(d_a) + a_k * stride_k + a_l;
      *global_b = reinterpret_cast<fvec4*>(d_b) + b_l * stride_l + b_k;
    }

  inline __device__
    void request(int stride_k, fvec4 **global_a, fvec4 thread_a[VectorsPerThreadX],
                 int stride_l, fvec4 **global_b, fvec4 thread_b[VectorsPerThreadX]) {
#pragma unroll
      for (int i = 0; i < VectorsPerThreadX; ++i) {
	thread_a[i] = (*global_a)[i * ThreadsPerBlockK * stride_k];
	thread_b[i] = (*global_b)[i * ThreadsPerBlockL * stride_l];
      }
      *global_a += (stride_k * ItemsPerBlockK);
      *global_b += VectorsPerBlockK;
    }

  inline __device__
    void commit(block_y_t (&block_a), fvec4 thread_a[VectorsPerThreadX],
                block_x_t (&block_b), fvec4 thread_b[VectorsPerThreadX]) {
      int a_k = threadIdx.x / VectorsPerBlockX;
      int a_l = threadIdx.x % VectorsPerBlockX;
      int b_k = threadIdx.x % VectorsPerBlockK;
      int b_l = threadIdx.x / VectorsPerBlockK;
#pragma unroll
      for (int i = 0; i < VectorsPerThreadX; ++i) {
	*reinterpret_cast<fvec4*>(&block_a[a_k + i * ThreadsPerBlockK][a_l * ItemsPerVectorX]) = thread_a[i];
#pragma unroll
	for (int j = 0; j < ItemsPerVectorX; ++j) {
	  block_b[b_k * ItemsPerVectorX + j][b_l + i * ThreadsPerBlockL] = thread_b[i].data[j];
	}
      }
    }

  inline __device__
    void store(float *ptr, const float &src) {
      asm volatile ("st.global.cg.f32 [%0], %1;\n"
		    : :
		    "l"(ptr),
		    "f"(src));
    }

  inline __device__
    static void gemm(float &d,
		     const float &a,
		     const float &b,
		     const float &c) {
      asm volatile ( "fma.rn.f32 %0, %1, %2, %3;\n"
		     : "=f"(d) : "f"(a), "f"(b), "f"(c));
    }

  inline __device__ void prefetch(block_y_t &block_a,
				  block_x_t &block_b,
				  fvec4 (&slice_a)[VectorsPerThreadY],
				  fvec4 (&slice_b)[VectorsPerThreadX],
				  int offset_y,
				  int offset_x,
				  int offset_k) {
    for (int i = 0; i < VectorsPerThreadX; ++i) {
      slice_b[i] = *reinterpret_cast<const fvec4*>(&block_b[offset_k][offset_x + (i * ThreadsPerWarpX * ItemsPerVectorX)]);
    }
    for (int i = 0; i < VectorsPerThreadY; ++i) {
      slice_a[i] = *reinterpret_cast<const fvec4*>(&block_a[offset_k][offset_y + (i * ThreadsPerWarpY * ItemsPerVectorY)]);
    }
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
    int offset_y = lane_y * ItemsPerVectorY + warp_y * ItemsPerWarpY;
    int offset_x = lane_x * ItemsPerVectorX + warp_x * ItemsPerWarpX;
    fvec4 slice_a[2][VectorsPerThreadY];
    fvec4 slice_b[2][VectorsPerThreadX];
    float tile_c[ItemsPerThreadY][ItemsPerThreadX];

    fvec4 *global_a;
    fvec4 *global_b;
    int stride_k;
    int stride_l;
    fvec4 thread_a[VectorsPerThreadX];
    fvec4 thread_b[VectorsPerThreadX];
    __shared__ block_y_t block_a;
    __shared__ block_x_t block_b;

    init(d_a, dim_m, ItemsPerBlockY * blockIdx.x, stride_k, &global_a,
         d_b, dim_k, ItemsPerBlockX * blockIdx.y, stride_l, &global_b);
    request(stride_k, &global_a, thread_a, stride_l, &global_b, thread_b);
    commit(block_a, thread_a, block_b, thread_b);
    __syncthreads();
#pragma unroll
    for (int y = 0; y < ItemsPerThreadY; ++y) {
#pragma unroll
      for (int x = 0; x < ItemsPerThreadX; ++x)
      {
	tile_c[y][x] = float(0);
      }
    }
    prefetch(block_a,
	     block_b,
	     slice_a[0],
	     slice_b[0],
	     offset_y,
	     offset_x,
	     0);
#pragma unroll
    for (int k = 0; k < dim_k; k += ItemsPerBlockK) {
#pragma unroll
      for (int offset_k = 0; offset_k < ItemsPerBlockK; offset_k++) {
	if ((offset_k == ItemsPerBlockK - 1) && k < dim_k-ItemsPerBlockK) {
	  __syncthreads();
	  commit(block_a, thread_a, block_b, thread_b);
	  __syncthreads();
	}
	if ((offset_k == 0) && dim_k-ItemsPerBlockK) {
	  request(stride_k, &global_a, thread_a, stride_l, &global_b, thread_b);
	}
	prefetch(block_a,
		 block_b,
		 slice_a[(offset_k + 1) % 2],
		 slice_b[(offset_k + 1) % 2],
		 offset_y,
		 offset_x,
		 (offset_k + 1) % ItemsPerBlockK);
	tile_a_t &tile_a = reinterpret_cast<tile_a_t&>(slice_a[(offset_k) % 2]);
	tile_b_t &tile_b = reinterpret_cast<tile_b_t&>(slice_b[(offset_k) % 2]);
#pragma unroll
	for (int y = 0; y < ItemsPerThreadY; ++y) {
#pragma unroll
	  for (int x = 0; x < ItemsPerThreadX; ++x) {
	    gemm(tile_c[y][x], tile_a[y], tile_b[x], tile_c[y][x]);
	  }
	}
      }
    }
#pragma unroll
    for (int ix = 0; ix < ItemsPerThreadX; ++ix) {
#pragma unroll
      for (int iy = 0; iy < ItemsPerThreadY; iy += ItemsPerVectorY) {
	int vx = ix / ItemsPerVectorX;
	int vy = iy / ItemsPerVectorY;
	int tx = offset_x + (vx * ThreadsPerWarpX * ItemsPerVectorX) + (ix % ItemsPerVectorX);
	int ty = offset_y + (vy * ThreadsPerWarpY * ItemsPerVectorY) + (iy % ItemsPerVectorY);
	int bx = ItemsPerBlockX * blockIdx.y + tx;
	int by = ItemsPerBlockY * blockIdx.x + ty;
#pragma unroll
	for (int i = 0; i < ItemsPerVectorY; ++i) {
	  int c_idx = bx * dim_m + by + i;
	  float slice_c = float(0);
	  if (bx < dim_n && (by + i) < dim_m) {
	    slice_c += tile_c[iy + i][ix];
	    store(d_c + c_idx, slice_c);
	  }
	}
      }
    }
  }
} // namespace cutlass
