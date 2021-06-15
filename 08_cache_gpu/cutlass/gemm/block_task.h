#pragma once

#include <stdint.h>

#include "../util/util.h"

#include "block_loader_crosswise.h"
#include "block_loader_congruous.h"
#include "thread_accumulator.h"

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

      typedef thread_accumulator<ItemsPerThreadY,ItemsPerThreadX> thread_accumulator_t;
      typedef io_vector<float, ItemsPerVectorY> lds_vector_a_t;
      typedef io_vector<float, ItemsPerVectorX> lds_vector_b_t;

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
      lds_vector_a_t local_slices_a[2][VectorsPerThreadY];
      lds_vector_b_t local_slices_b[2][VectorsPerThreadX];
      block_loader_a_t loader_a;
      block_loader_b_t loader_b;
      thread_accumulator_t accumulator;

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
	  loader_b(d_b, dim_k, ItemsPerBlockX * blockIdx.y),
	  accumulator()
	  {}

      inline __device__ void request_local_prefetch(lds_vector_a_t (&slice_a)[VectorsPerThreadY],
						    lds_vector_b_t (&slice_b)[VectorsPerThreadX],
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
	  slice_b[i].load(&scratch->block_b[offset_k][offset_x + (i * ThreadsPerWarpX * ItemsPerVectorX)]);
	}
	for (int i = 0; i < VectorsPerThreadY; ++i) {
	  slice_a[i].load(&scratch->block_a[offset_k][offset_y + (i * ThreadsPerWarpY * ItemsPerVectorY)]);
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
	  accumulator.init();
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
	      accumulator.multiply_accumulate(tile_a, tile_b);
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
	    accumulator.multiply_accumulate(tile_a, tile_b);
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
		  c_slice = alpha * accumulator.get(ix, iy + i) + beta * c_slice;
		  stg_cg(d_c + c_idx, c_slice);
		}
	      }
	    }
	  }
	}
    };


  } // namespace gemm
} // namespace cutlass
