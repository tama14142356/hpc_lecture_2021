#pragma once
#include <stdint.h>
#include "../util/util.h"
#include "block_task.h"
#include "grid_raster.h"

namespace cutlass {
namespace gemm {

  __global__ void kernel(
                       int m,                      ///< Height in rows of op(A) and C
                       int n,                      ///< Width in columns of op(B) and C
                       int k,                      ///< Width in columns of op(A) and height in rows of op(B)
                       float *d_a,               ///< Pointer to matrix A array values
                       float *d_b,               ///< Pointer to matrix B array values
                       float *d_c)               ///< Pointer to matrix C array values
{
  typedef block_task<
    float,
    float,
    16,
    16,
    4,
    false> block_task_t;

    // Declare statically-allocated shared storage
    __shared__ typename block_task_t::scratch_storage_t smem;

    // Construct and run the task
    block_task_t(
        &smem,
        d_a,
        d_b,
        d_c,
        m,
        n,
        k).run();
}


void dispatch(
    int             m,                              ///< Height in rows of op(A) and C
    int             n,                              ///< Width in columns of op(B) and C
    int             k,                              ///< Width in columns of op(A) and height in rows of op(B)
    float           alpha,
    float           beta,
    float         *d_a,                           ///< Device pointer to matrix A array values
    float         *d_b,                           ///< Device pointer to matrix B array values
    float         *d_c)                           ///< Device pointer to matrix C array values
{
  static const matrix_transform_t::kind_t TransformA = matrix_transform_t::NonTranspose;
  static const matrix_transform_t::kind_t TransformB = matrix_transform_t::NonTranspose;
  typedef grid_raster<
    64,
    64,
    TransformA,
    TransformB,
    grid_raster_strategy::Default>
    grid_raster_t;
  dim3 block = dim3(64);
  dim3 grid = grid_raster_t::grid_dims(m, n);

  gemm::kernel<<< grid, block >>>(
               m,
               n,
               k,
               d_a,
               d_b,
               d_c);
}


} // namespace gemm
} // namespace cutlass
