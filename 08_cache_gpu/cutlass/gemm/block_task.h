#pragma once

#include <stdint.h>

#include "../util/util.h"

#include "grid_raster.h"
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

      typedef thread_accumulator<ItemsPerThreadY,ItemsPerThreadX,float,float> thread_accumulator_t;
      typedef io_vector<float, ItemsPerVectorY> lds_vector_a_t;
      typedef io_vector<float, ItemsPerVectorX> lds_vector_b_t;

      typedef grid_raster<
	ItemsPerBlockY,
	ItemsPerBlockX,
	matrix_transform_t::NonTranspose,
	matrix_transform_t::NonTranspose,
	grid_raster_strategy::Default>
	  grid_raster_t;

      struct scratch_storage_t {
	float __align__(16) block_a[ItemsPerBlockK][ItemsPerBlockY];
	float __align__(16) block_b[ItemsPerBlockK][ItemsPerBlockX];
	typename thread_accumulator_t::scratch_storage_t accum_scratch;
      };

      scratch_storage_t *scratch;
      float *d_c;
      int dim_m;
      int dim_n;
      int dim_k;
      grid_raster_t grid_raster;
      int thread_strip_offset_a;
      int thread_strip_offset_b;
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
		   int dim_k)
	:
	  scratch(scratch),
	  d_c(d_c),
	  dim_m(dim_m),
	  dim_n(dim_n),
	  dim_k(dim_k),

	  loader_a(
		   d_a,
		   dim_m,
		   ItemsPerBlockY * blockIdx.x),

	  loader_b(
		   d_b,
		   dim_k,
		   ItemsPerBlockX * blockIdx.y),

	  accumulator(scratch->accum_scratch)
	  {}


      inline __device__ void request_local_prefetch(
						    lds_vector_a_t (&slice_a)[VectorsPerThreadY],  ///< Slice from A
						    lds_vector_b_t (&slice_b)[VectorsPerThreadX],  ///< Slice from B
						    int tile_offset_k)
      {
	int warp_id = threadIdx.x / ThreadsPerWarp;
	int warp_x = warp_id % WarpsPerBlockX;
	int warp_y = warp_id / WarpsPerBlockX;
	int lane_id = threadIdx.x % ThreadsPerWarp;
	int lane_x = lane_id / ThreadsPerWarpY;
	int lane_y = lane_id % ThreadsPerWarpY;
	thread_strip_offset_a = lane_y * ItemsPerVectorY + warp_y * ItemsPerWarpY;
	thread_strip_offset_b = lane_x * ItemsPerVectorX + warp_x * ItemsPerWarpX;
	// Load B strip
	for (int i = 0; i < VectorsPerThreadX; ++i)
	{
	  slice_b[i].load(
			  &scratch->block_b[tile_offset_k][thread_strip_offset_b + (i * ThreadsPerWarpX * ItemsPerVectorX)]);
	}

	// Load A strip
	for (int i = 0; i < VectorsPerThreadY; ++i)
	{
	  slice_a[i].load(
			  &scratch->block_a[tile_offset_k][thread_strip_offset_a + (i * ThreadsPerWarpY * ItemsPerVectorY)]);
	}
      }


      //-------------------------------------------------------------------------
      // Epilogue
      //-------------------------------------------------------------------------

      /**
       * Performs the GEMM epilogue:
       *   - Applies the scalar multipliers and addends to the accumulators
       *   - Write the result to the output matrix
       */
      __forceinline__ __device__
	void epilogue()
	{
	  float alpha = 1.0;
	  float beta = 0.0;
#pragma unroll
	  for (int x = 0; x < ItemsPerThreadX; ++x)
	  {
#pragma unroll
	    for (int y = 0; y < ItemsPerThreadY; y += ItemsPerVectorY)
	    {
	      int thread_strip_b = x / ItemsPerVectorX;
	      int thread_strip_a = y / ItemsPerVectorY;

	      int thread_item_coords_tile_x = thread_strip_offset_b + (thread_strip_b * ThreadsPerWarpX * ItemsPerVectorX) + (x % ItemsPerVectorX);
	      int thread_item_coords_tile_y = thread_strip_offset_a + (thread_strip_a * ThreadsPerWarpY * ItemsPerVectorY) + (y % ItemsPerVectorY);

	      int c_idx = (grid_raster.block_item_coords.x + thread_item_coords_tile_x) * dim_m +
		grid_raster.block_item_coords.y + thread_item_coords_tile_y;

	      float *my_c = d_c + c_idx;

#pragma unroll
	      for (int i = 0; i < ItemsPerVectorY; ++i)
	      {
		float c_slice = float(0);
		float *c_ptr = my_c + i;

		if ((grid_raster.block_item_coords.x + thread_item_coords_tile_x) < dim_n &&
		    (grid_raster.block_item_coords.y + thread_item_coords_tile_y + i) < dim_m)
		{
		  c_slice = alpha * accumulator.get(x, y + i) + beta * c_slice;

		  stg_cg(c_ptr, c_slice);
		}
	      }
	    }
	  }
	}


      //-------------------------------------------------------------------------
      // Tile consumption
      //-------------------------------------------------------------------------

      /**
       * Consume a tile of A and B each
       */
      template <bool DoGlobalPrefetch>
	__forceinline__ __device__
	void consume_tile()
	{
	  // Unroll ItemsPerBlockK iterations of outer-product accumulations
#pragma unroll
	  for (int tile_offset_k = 0; tile_offset_k < ItemsPerBlockK; tile_offset_k += 1)
	  {
	    // Last strip commits global prefetch for next tile
	    if ((tile_offset_k == ItemsPerBlockK - 1) && DoGlobalPrefetch)
	    {
	      __syncthreads();
	      // Commit global prefetch data to scratch page
	      loader_a.commit(scratch->block_a);
	      loader_b.commit(scratch->block_b);

	      __syncthreads();
	    }

	    // Request local prefetch for next strip
	    request_local_prefetch(
				   local_slices_a[(tile_offset_k + 1) % 2],
				   local_slices_b[(tile_offset_k + 1) % 2],
				   (tile_offset_k + 1) % ItemsPerBlockK);

	    // Request global prefetch for next tile on first strip
	    if ((tile_offset_k == 0) && DoGlobalPrefetch)
	    {
	      loader_b.request();
	      loader_a.request();
	    }

	    // Cast strip-mined loads to contiguous array of float
	    typedef float thread_tile_a_t[VectorsPerThreadY * ItemsPerVectorY];
	    typedef float thread_tile_b_t[VectorsPerThreadX * ItemsPerVectorX];
	    thread_tile_a_t &thread_tile_a = reinterpret_cast<thread_tile_a_t&>(local_slices_a[(tile_offset_k) % 2]);
	    thread_tile_b_t &thread_tile_b = reinterpret_cast<thread_tile_b_t&>(local_slices_b[(tile_offset_k) % 2]);

	    // Accumulate this dp-stripe product
	    accumulator.multiply_accumulate(thread_tile_a, thread_tile_b);
	  }
	}


      //-------------------------------------------------------------------------
      // GEMM API
      //-------------------------------------------------------------------------

      /**
       * Compute GEMM
       */
      __forceinline__ __device__
	void run()
	{
	  int block_item_coords_k = 0;

	  // Request global prefetch of first tile
	  loader_a.request();
	  loader_b.request();

	  // Commit global prefetch of first tile to shared memory
	  loader_a.commit(scratch->block_a);
	  loader_b.commit(scratch->block_b);

	  // Advance to next A,B tiles in K-axis
	  block_item_coords_k += ItemsPerBlockK;

	  // Synchronize shared tiles and prepared accumulator
	  __syncthreads();

	  // Initialize thread's slice of accumulators
	  accumulator.init();

	  // Request first iteration of local prefetch strips
	  request_local_prefetch(
				 local_slices_a[0],
				 local_slices_b[0],
				 0);

	  //
	  // Main loop
	  //

	  // Consume tiles in A and B along the K-axis (all but last tile)
#pragma unroll 1
	  while (block_item_coords_k < dim_k)
	  {
	    consume_tile<true>();

	    // Advance to next A,B tiles in K-axis
	    block_item_coords_k += ItemsPerBlockK;
	  }

	  // Consume last tile
	  consume_tile<false>();

	  //
	  // Eplilogue
	  //

	  epilogue();
	}
    };


  } // namespace gemm
} // namespace cutlass
