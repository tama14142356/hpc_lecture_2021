#pragma once
#include <stdint.h>
#include "../util/util.h"
#include "block_task.h"
#include "grid_raster.h"
#include "k_split_control.h"

namespace cutlass {
namespace gemm {

  template <typename epilogue_op_t>
  __global__ void kernel(
                       int m,                      ///< Height in rows of op(A) and C
                       int n,                      ///< Width in columns of op(B) and C
                       int k,                      ///< Width in columns of op(A) and height in rows of op(B)
                       k_split_control k_split,    ///< Abstraction for controlling inter-block k-splitting
                       epilogue_op_t op,           ///< Epilogue operation to update matrix C
                       float *d_a,               ///< Pointer to matrix A array values
                       float *d_b,               ///< Pointer to matrix B array values
                       float *d_c)               ///< Pointer to matrix C array values
{
  typedef block_task<
    float,
    float,
    16,
    16,
    epilogue_op_t,
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
        op,
        m,
        n,
        k,
        k_split).run();
}


template <typename epilogue_op_t>
void dispatch(
    int             m,                              ///< Height in rows of op(A) and C
    int             n,                              ///< Width in columns of op(B) and C
    int             k,                              ///< Width in columns of op(A) and height in rows of op(B)
    float           alpha,
    float           beta,
    float         *d_a,                           ///< Device pointer to matrix A array values
    float         *d_b,                           ///< Device pointer to matrix B array values
    float         *d_c,                           ///< Device pointer to matrix C array values
    cudaStream_t    stream = 0,                     ///< CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
    bool            debug_synchronous = true)       ///< Whether or not to synchronize the stream after every kernel launch
                                                    ///  to check for errors.  Also causes launch configurations to be printed
                                                    ///  to the console if DEBUG is defined.  Default is \p false.
{
  static const matrix_transform_t::kind_t TransformA = matrix_transform_t::NonTranspose;
  static const matrix_transform_t::kind_t TransformB = matrix_transform_t::NonTranspose;
  epilogue_op_t epilogue(alpha, beta);
  typedef grid_raster<
    64,
    64,
    TransformA,
    TransformB,
    grid_raster_strategy::Default>
    grid_raster_t;
  int dynamic_smem_bytes = 0;
  int max_sm_occupancy = 8;
  dim3 block = dim3(64);
  dim3 grid = grid_raster_t::grid_dims(m, n);
  int sm_count;
  get_sm_count(sm_count);
  int *d_flags;
  cudaGetSymbolAddress((void**) &d_flags, d_flags_split_k);

  k_split_control k_split(
                          d_flags,
                          sm_count,
                          max_sm_occupancy,
                          k,
                          8,
                          block,
                          grid);
  gemm::kernel<epilogue_op_t>
    <<< grid,
    block,
    dynamic_smem_bytes,
    stream >>>(
               m,
               n,
               k,
               k_split,
               epilogue,
               d_a,
               d_b,
               d_c);
}


} // namespace gemm
} // namespace cutlass
