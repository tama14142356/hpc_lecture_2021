#pragma once

#include <stdint.h>
#include "io_intrinsics.h"
#include "block_loader_crosswise.h"
#include "block_loader_congruous.h"

namespace cutlass {
  namespace gemm {

    struct block_task
    {
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

      typedef io_vector matrix_t;

      struct scratch_storage_t {
	float __align__(16) block_a[ItemsPerBlockK][ItemsPerBlockY];
	float __align__(16) block_b[ItemsPerBlockK][ItemsPerBlockX];
      };

      scratch_storage_t *scratch;
      float *d_c;
      int dim_m;
      int dim_n;
      int dim_k;
      int offset_y;
      int offset_x;
      matrix_t local_slices_a[2][VectorsPerThreadY];
      matrix_t local_slices_b[2][VectorsPerThreadX];
      block_loader_a_t loader_a;
      block_loader_b_t loader_b;
      float accumulators[ItemsPerThreadY][ItemsPerThreadX];

      inline __device__
	static void mad(float &d,
			const float &a,
			const float &b,
			const float &c) {
	  asm volatile ( "fma.rn.f32 %0, %1, %2, %3;\n"
			 : "=f"(d) : "f"(a), "f"(b), "f"(c));
	}

      inline __device__
	void mad_xy(float (&tile_a)[ItemsPerThreadY],
		    float (&tile_b)[ItemsPerThreadX],
		    int x,
		    int y) {
	  mad(
	      accumulators[y][x],
	      tile_a[y],
	      tile_b[x],
	      accumulators[y][x]);
	}

      inline __device__
	block_task(
		   scratch_storage_t *scratch,
		   float *d_a,
		   float *d_b,
		   float *d_c,
		   int dim_m,
		   int dim_n,
		   int dim_k) :
	  scratch(scratch),
	  d_c(d_c),
	  dim_m(dim_m),
	  dim_n(dim_n),
	  dim_k(dim_k),
	  loader_a(d_a, dim_m, ItemsPerBlockY * blockIdx.x),
	  loader_b(d_b, dim_k, ItemsPerBlockX * blockIdx.y) {}

      inline __device__ void request_local_prefetch(matrix_t (&slice_a)[VectorsPerThreadY],
						    matrix_t (&slice_b)[VectorsPerThreadX],
						    int offset_k)
      {
	int warp_id = threadIdx.x / ThreadsPerWarp;
	int warp_x = warp_id % WarpsPerBlockX;
	int warp_y = warp_id / WarpsPerBlockX;
	int lane_id = threadIdx.x % ThreadsPerWarp;
	int lane_x = lane_id / ThreadsPerWarpY;
	int lane_y = lane_id % ThreadsPerWarpY;
	offset_y = lane_y * ItemsPerVectorY + warp_y * ItemsPerWarpY;
	offset_x = lane_x * ItemsPerVectorX + warp_x * ItemsPerWarpX;
	for (int i = 0; i < VectorsPerThreadX; ++i) {
	  slice_b[i] = *reinterpret_cast<const matrix_t*>(&scratch->block_b[offset_k][offset_x + (i * ThreadsPerWarpX * ItemsPerVectorX)]);
	}
	for (int i = 0; i < VectorsPerThreadY; ++i) {
	  slice_a[i] = *reinterpret_cast<const io_vector*>(&scratch->block_a[offset_k][offset_y + (i * ThreadsPerWarpY * ItemsPerVectorY)]);
	}
      }

      __forceinline__ __device__
	void run()
	{
	  int block_item_coords_k = 0;
	  loader_a.request();
	  loader_b.request();
	  loader_a.commit(scratch->block_a);
	  loader_b.commit(scratch->block_b);
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
	  request_local_prefetch(local_slices_a[0],
				 local_slices_b[0],
				 0);
#pragma unroll 1
	  while (block_item_coords_k < dim_k) {
#pragma unroll
	    for (int offset_k = 0; offset_k < ItemsPerBlockK; offset_k += 1) {
	      if ((offset_k == ItemsPerBlockK - 1)) {
		__syncthreads();
		loader_a.commit(scratch->block_a);
		loader_b.commit(scratch->block_b);
		__syncthreads();
	      }
	      request_local_prefetch(local_slices_a[(offset_k + 1) % 2],
				     local_slices_b[(offset_k + 1) % 2],
				     (offset_k + 1) % ItemsPerBlockK);
	      if ((offset_k == 0)) {
		loader_b.request();
		loader_a.request();
	      }
	      typedef float tile_a_t[VectorsPerThreadY * ItemsPerVectorY];
	      typedef float tile_b_t[VectorsPerThreadX * ItemsPerVectorX];
	      tile_a_t &tile_a = reinterpret_cast<tile_a_t&>(local_slices_a[(offset_k) % 2]);
	      tile_b_t &tile_b = reinterpret_cast<tile_b_t&>(local_slices_b[(offset_k) % 2]);
#pragma unroll
	      for (int y = 0; y < ItemsPerThreadY; ++y) {
#pragma unroll
		for (int x = 0; x < ItemsPerThreadX; ++x) {
		  mad_xy(tile_a, tile_b, x, y);
		}
	      }
	    }
	    block_item_coords_k += ItemsPerBlockK;
	  }
#pragma unroll
	  for (int offset_k = 0; offset_k < ItemsPerBlockK; offset_k += 1) {
	    request_local_prefetch(local_slices_a[(offset_k + 1) % 2],
				   local_slices_b[(offset_k + 1) % 2],
				   (offset_k + 1) % ItemsPerBlockK);
	    typedef float tile_a_t[VectorsPerThreadY * ItemsPerVectorY];
	    typedef float tile_b_t[VectorsPerThreadX * ItemsPerVectorX];
	    tile_a_t &tile_a = reinterpret_cast<tile_a_t&>(local_slices_a[(offset_k) % 2]);
	    tile_b_t &tile_b = reinterpret_cast<tile_b_t&>(local_slices_b[(offset_k) % 2]);
#pragma unroll
	    for (int y = 0; y < ItemsPerThreadY; ++y) {
#pragma unroll
	      for (int x = 0; x < ItemsPerThreadX; ++x) {
		mad_xy(tile_a, tile_b, x, y);
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
		  stg_cg(d_c + c_idx, c_slice);
		}
	      }
	    }
	  }
	}
    };


  } // namespace gemm
} // namespace cutlass
