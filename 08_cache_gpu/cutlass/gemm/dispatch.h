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
  __shared__ typename block_task::scratch_storage_t smem;
  block_task(
        &smem,
        d_a,
        d_b,
        d_c,
        m,
        n,
        k).run();
}
