#pragma once

#include <stdint.h>

#include "../util/util.h"

#include "grid_raster.h"
#include "block_loader.h"
#include "thread_accumulator.h"

namespace cutlass {
namespace gemm {

struct block_task
{
    enum
    {
        ItemsPerVectorY = 4,
        ItemsPerVectorX = 4,
        VectorsPerThreadY = 2,
        VectorsPerThreadX = 2,
        ThreadsPerWarp = 32,
        ThreadsPerWarpY = 4,
        ThreadsPerWarpX = ThreadsPerWarp / ThreadsPerWarpY,
        WarpsPerBlockY = 2,
        WarpsPerBlockX = 1,
        ItemsPerThreadY = VectorsPerThreadY * ItemsPerVectorY,
        ItemsPerThreadX = VectorsPerThreadX * ItemsPerVectorX,
        ItemsPerWarpY = ThreadsPerWarpY * ItemsPerThreadY,
        ItemsPerWarpX = ThreadsPerWarpX * ItemsPerThreadX,
        ItemsPerBlockY = WarpsPerBlockY * ItemsPerWarpY,
        ItemsPerBlockX = WarpsPerBlockX * ItemsPerWarpX,
        ItemsPerBlockK = 8,
    };

    /// Accumulator type
    typedef thread_accumulator<
            ItemsPerThreadY,
            ItemsPerThreadX,
            float,
            float>
        thread_accumulator_t;

    /// Load-from-shared data movement type for A-tile, coarsened by ItemsPerVectorY
    typedef io_vector<float, ItemsPerVectorY> lds_vector_a_t;

    /// Load-from-shared data movement type for B-tile, coarsened by ItemsPerVectorX
    typedef io_vector<float, ItemsPerVectorX> lds_vector_b_t;

    /// Thread block rasterization helper type
    typedef grid_raster<
      ItemsPerBlockY,
      ItemsPerBlockX,
      matrix_transform_t::NonTranspose,
      matrix_transform_t::NonTranspose,
      grid_raster_strategy::Default>
    grid_raster_t;


    /// Tile loader type for matrix A
    typedef block_loader<
      ItemsPerBlockY,
      ItemsPerBlockK,
      ItemsPerBlockX,
      16,
      load_algorithm::CongruousCopy>
    block_loader_a_t;


    /// Tile loader type for matrix B
    typedef block_loader<
      ItemsPerBlockY,
      ItemsPerBlockK,
      ItemsPerBlockX,
      16,
      load_algorithm::CrosswiseCopy>
    block_loader_b_t;

    /// Shared memory layout for scratch storage
    struct scratch_storage_t
    {
        float __align__(16) block_a[ItemsPerBlockK][ItemsPerBlockY];
        float __align__(16) block_b[ItemsPerBlockK][ItemsPerBlockX];
        typename thread_accumulator_t::scratch_storage_t accum_scratch;
    };

    /// Scratch storage reference
    scratch_storage_t *scratch;

    /// Pointer to matrix C
    float *d_c;

    /// Matrix height in rows of trans_op(A) and C
    int dim_m;

    /// Matrix width in columns of trans_op(B) and C
    int dim_n;

    /// Thread block's base value_t coordinates (m, n) in matrix C
    grid_raster_t grid_raster;

    /// Thread block's current coordinate (k) within A|B matrices
    int block_item_coords_k;

    /// Thread block's ending coordinate (k) within A|B matrices (one-past)
    int block_end_item_k;

    /// Warp's coordinates (x, y) in thread block
    int2 block_warp_coords;

    /// Thread's coordinates (x, y) in warp
    int2 warp_thread_coords;

    /// Thread's base item offset within strip of A tile
    int thread_strip_offset_a;

    /// Thread's base item offset within strip of B tile
    int thread_strip_offset_b;

    /// Thread's active-k/prefetch-k slices from shared A tile
    lds_vector_a_t local_slices_a[2][VectorsPerThreadY];

    /// Thread's active-k/prefetch-k slices from shared B tile
    lds_vector_b_t local_slices_b[2][VectorsPerThreadX];

    /// A tile loader
    block_loader_a_t loader_a;

    /// B tile loader
    block_loader_b_t loader_b;

    /// C tile accumulator
    thread_accumulator_t accumulator;


    //-------------------------------------------------------------------------
    // Coordinate system helpers
    //-------------------------------------------------------------------------

    /// Compute the warp's coordinates (x, y) in thread block
    inline __device__
    int2 warp_coords()
    {
        int warp_id = threadIdx.x / ThreadsPerWarp;
        return make_int2(
            warp_id % WarpsPerBlockX,
            warp_id / WarpsPerBlockX);
    }


    /// Compute the thread's lane-coordinates (x, y) in warp
    inline __device__
    int2 thread_coords()
    {
        int lane_id = threadIdx.x % ThreadsPerWarp;

        // Maxwell+ mapping of threads within a 2D warp for maximal LDS bandwidth
        return make_int2(
            lane_id / ThreadsPerWarpY,
            lane_id % ThreadsPerWarpY);
    }


    //-------------------------------------------------------------------------
    // Constructor API
    //-------------------------------------------------------------------------

    /// Constructor
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
        block_item_coords_k(0),
        block_end_item_k(dim_k),
        block_warp_coords(warp_coords()),
        warp_thread_coords(thread_coords()),
        thread_strip_offset_a((warp_thread_coords.y * ItemsPerVectorY) + (block_warp_coords.y * ItemsPerWarpY)),
        thread_strip_offset_b((warp_thread_coords.x * ItemsPerVectorX) + (block_warp_coords.x * ItemsPerWarpX)),

        loader_a(
            d_a,                                                            // d_matrix
            dim_m,                                                          // matrix_values_l
            dim_m,
            1,
            ItemsPerBlockY * blockIdx.x,
            dim_k),                                              // block_end_item_k

        loader_b(
            d_b,                                                            // d_matrix
            dim_n,
            1,
            dim_k,
            ItemsPerBlockX * blockIdx.y,
            dim_k),                                              // block_end_item_k

        accumulator(scratch->accum_scratch)
    {}


    //-------------------------------------------------------------------------
    // Prefetching utility methods
    //-------------------------------------------------------------------------

    /**
     * Request the calling thread's slices of the shared tiles at depth \p tile_offset_k
     */
    inline __device__ void request_local_prefetch(
        lds_vector_a_t (&slice_a)[VectorsPerThreadY],  ///< Slice from A
        lds_vector_b_t (&slice_b)[VectorsPerThreadX],  ///< Slice from B
        int tile_offset_k)
    {
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
        // Quit if the thread block is fully out-of-bounds
        if (grid_raster.is_block_oob(dim_m, dim_n))
        {
            asm volatile("exit;");
        }

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
        while (block_item_coords_k < block_end_item_k)
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
