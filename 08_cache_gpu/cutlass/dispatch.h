#pragma once
#include "block_task.h"
using namespace cutlass;
using namespace gemm;
__global__ void kernel(
                       int m,
                       int n,
                       int k,
                       float *d_a,
                       float *d_b,
                       float *d_c)
{
  __shared__ scratch_storage_t smem;
  run(
      &smem,
      d_a,
      d_b,
      d_c,
      m,
      n,
      k);
}
