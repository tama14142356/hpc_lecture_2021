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
  struct scratch_storage_t {
    block_y_t block_a;
    block_x_t block_b;
  };

  inline __device__
    void init_a(float *d_a, int dim_m, int block_offset, int &stride_k, fvec4 **global_a) {
      stride_k = dim_m / ItemsPerVectorX;
      int vector_l = threadIdx.x % VectorsPerBlockX;
      int tile_k = threadIdx.x / VectorsPerBlockX;
      int tile_l = vector_l + block_offset / ItemsPerVectorX;
      *global_a = reinterpret_cast<fvec4*>(d_a) + tile_k * stride_k + tile_l;
    }

  inline __device__
    void request_a(int stride_k, fvec4 **global_a, fvec4 thread_a[VectorsPerThreadX]) {
#pragma unroll
      for (int i = 0; i < VectorsPerThreadX; ++i) {
	thread_a[i] = (*global_a)[i * ThreadsPerBlockK * stride_k];
      }
      *global_a += (stride_k * ItemsPerBlockK);
    }

  inline __device__
    void commit_a(float (&scratch_tile)[ItemsPerBlockK][ItemsPerBlockY], fvec4 thread_a[VectorsPerThreadX]) {
      int vector_k = threadIdx.x / VectorsPerBlockX;
      int vector_l = threadIdx.x % VectorsPerBlockX;
#pragma unroll
      for (int i = 0; i < VectorsPerThreadX; ++i) {
	*reinterpret_cast<fvec4*>(&scratch_tile[vector_k + i * ThreadsPerBlockK][vector_l * ItemsPerVectorX]) =
	  thread_a[i];
      }
    }

  inline __device__
    void init_b(float *d_b, int dim_k, int block_offset, int &stride_l, fvec4 **global_b) {
      stride_l = dim_k / ItemsPerVectorX;
      int vector_l = threadIdx.x / VectorsPerBlockK;
      int tile_k = threadIdx.x % VectorsPerBlockK;
      int tile_l = vector_l + block_offset;
      *global_b = reinterpret_cast<fvec4*>(d_b) + tile_l * stride_l + tile_k;
    }

  inline __device__
    void request_b(int stride_l, fvec4 **global_b, fvec4 thread_b[VectorsPerThreadX]) {
#pragma unroll
      for (int i = 0; i < VectorsPerThreadX; ++i) {
	thread_b[i] = (*global_b)[i * ThreadsPerBlockL * stride_l];
      }
      *global_b += VectorsPerBlockK;
    }

    template <int ItemsPerBlockX>
      inline __device__
      void commit_b(float (&scratch_tile)[ItemsPerBlockK][ItemsPerBlockX], fvec4 thread_b[VectorsPerThreadX]) {
	int vector_k = threadIdx.x % VectorsPerBlockK;
	int vector_l = threadIdx.x / VectorsPerBlockK;
#pragma unroll
	for (int i = 0; i < VectorsPerThreadX; ++i) {
#pragma unroll
	  for (int j = 0; j < ItemsPerVectorX; ++j) {
	    scratch_tile[vector_k * ItemsPerVectorX + j][vector_l + i * ThreadsPerBlockL] = thread_b[i].data[j];
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
    static void mad(float &d,
		    const float &a,
		    const float &b,
		    const float &c) {
      asm volatile ( "fma.rn.f32 %0, %1, %2, %3;\n"
		     : "=f"(d) : "f"(a), "f"(b), "f"(c));
    }

  inline __device__
    void mad_xy(float C[ItemsPerThreadY][ItemsPerThreadX],
		float (&A)[ItemsPerThreadY],
		float (&B)[ItemsPerThreadX],
		int x,
		int y) {
      mad(C[y][x],
	  A[y],
	  B[x],
	  C[y][x]);
    }

  inline __device__ void request_local_prefetch(scratch_storage_t *scratch,
						fvec4 (&slice_a)[VectorsPerThreadY],
						fvec4 (&slice_b)[VectorsPerThreadX],
						int offset_y,
						int offset_x,
						int offset_k) {
    for (int i = 0; i < VectorsPerThreadX; ++i) {
      slice_b[i] = *reinterpret_cast<const fvec4*>(&scratch->block_b[offset_k][offset_x + (i * ThreadsPerWarpX * ItemsPerVectorX)]);
    }
    for (int i = 0; i < VectorsPerThreadY; ++i) {
      slice_a[i] = *reinterpret_cast<const fvec4*>(&scratch->block_a[offset_k][offset_y + (i * ThreadsPerWarpY * ItemsPerVectorY)]);
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
    fvec4 local_slices_a[2][VectorsPerThreadY];
    fvec4 local_slices_b[2][VectorsPerThreadX];
    float accumulators[ItemsPerThreadY][ItemsPerThreadX];

    int block_item_coords_k = 0;
    fvec4 *global_a;
    fvec4 *global_b;
    int stride_k;
    int stride_l;
    fvec4 thread_a[VectorsPerThreadX];
    fvec4 thread_b[VectorsPerThreadX];

    init_a(d_a, dim_m, ItemsPerBlockY * blockIdx.x, stride_k, &global_a);
    init_b(d_b, dim_k, ItemsPerBlockX * blockIdx.y, stride_l, &global_b);
    request_a(stride_k, &global_a, thread_a);
    request_b(stride_l, &global_b, thread_b);
    __shared__ scratch_storage_t scratch;
    commit_a(scratch.block_a, thread_a);
    commit_b(scratch.block_b, thread_b);
    block_item_coords_k += ItemsPerBlockK;
    __syncthreads();
#pragma unroll
    for (int y = 0; y < ItemsPerThreadY; ++y) {
#pragma unroll
      for (int x = 0; x < ItemsPerThreadX; ++x)
      {
	accumulators[y][x] = float(0);
      }
    }
    request_local_prefetch(&scratch,
			   local_slices_a[0],
			   local_slices_b[0],
			   offset_y,
			   offset_x,
			   0);
#pragma unroll 1
    while (block_item_coords_k < dim_k) {
#pragma unroll
      for (int offset_k = 0; offset_k < ItemsPerBlockK; offset_k += 1) {
	if ((offset_k == ItemsPerBlockK - 1)) {
	  __syncthreads();
	  commit_a(scratch.block_a, thread_a);
	  commit_b(scratch.block_b, thread_b);
	  __syncthreads();
	}
	request_local_prefetch(&scratch,
			       local_slices_a[(offset_k + 1) % 2],
			       local_slices_b[(offset_k + 1) % 2],
			       offset_y,
			       offset_x,
			       (offset_k + 1) % ItemsPerBlockK);
	if ((offset_k == 0)) {
	  request_a(stride_k, &global_a, thread_a);
	  request_b(stride_l, &global_b, thread_b);
	}
	typedef float tile_a_t[VectorsPerThreadY * ItemsPerVectorY];
	typedef float tile_b_t[VectorsPerThreadX * ItemsPerVectorX];
	tile_a_t &tile_a = reinterpret_cast<tile_a_t&>(local_slices_a[(offset_k) % 2]);
	tile_b_t &tile_b = reinterpret_cast<tile_b_t&>(local_slices_b[(offset_k) % 2]);
#pragma unroll
	for (int y = 0; y < ItemsPerThreadY; ++y) {
#pragma unroll
	  for (int x = 0; x < ItemsPerThreadX; ++x) {
	    mad_xy(accumulators, tile_a, tile_b, x, y);
	  }
	}
      }
      block_item_coords_k += ItemsPerBlockK;
    }
#pragma unroll
    for (int offset_k = 0; offset_k < ItemsPerBlockK; offset_k += 1) {
      request_local_prefetch(&scratch,
			     local_slices_a[(offset_k + 1) % 2],
			     local_slices_b[(offset_k + 1) % 2],
			     offset_y,
			     offset_x,
			     (offset_k + 1) % ItemsPerBlockK);
      typedef float tile_a_t[VectorsPerThreadY * ItemsPerVectorY];
      typedef float tile_b_t[VectorsPerThreadX * ItemsPerVectorX];
      tile_a_t &tile_a = reinterpret_cast<tile_a_t&>(local_slices_a[(offset_k) % 2]);
      tile_b_t &tile_b = reinterpret_cast<tile_b_t&>(local_slices_b[(offset_k) % 2]);
#pragma unroll
      for (int y = 0; y < ItemsPerThreadY; ++y) {
#pragma unroll
	for (int x = 0; x < ItemsPerThreadX; ++x) {
	  mad_xy(accumulators, tile_a, tile_b, x, y);
	}
      }
    }
    float alpha = 1.0;
    float beta = 0.0;
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
	  float c_slice = float(0);
	  if (bx < dim_n && (by + i) < dim_m) {
	    c_slice = alpha * accumulators[iy + i][ix] + beta * c_slice;
	    store(d_c + c_idx, c_slice);
	  }
	}
      }
    }
  }
} // namespace cutlass
