#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#define DEBUG

#include "cutlass.h"
#include "timer.h"
using namespace cutlass;

int main(int argc, const char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 8192;
  float alpha = 1.0;
  float beta = 0.0;
  typedef float value_t;
  typedef float accum_t;
  int g_timing_iterations = 10;
  float *A, *B, *C, *C2;
  cudaMallocManaged(&A, m * k * sizeof(float));
  cudaMallocManaged(&B, k * n * sizeof(float));
  cudaMallocManaged(&C, m * n * sizeof(float));
  cudaMallocManaged(&C2, m * n * sizeof(float));
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A[k*i+j] = drand48();
  for (int i=0; i<k; i++)
    for (int j=0; j<n; j++)
      B[n*i+j] = drand48();
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      C[m*i+j] = drand48();
      C2[m*i+j] = drand48();
    }
  }
  cublasHandle_t g_cublas_handle;
  cublasCreate(&g_cublas_handle);
  gpu_timer timer;
  for (int i = 0; i < g_timing_iterations+2; i++) {
    if (i == 2) timer.start();
    cublasSgemm(g_cublas_handle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		m,
		n,
		k,
		&alpha,
		A,
		m,
		B,
		k,
		&beta,
		C,
		m);
  }
  timer.stop();
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = timer.elapsed_millis() / g_timing_iterations;
  double cublas_flops = double(num_flops) / tcublas / 1.0e6;
  int tile = 64;
  dim3 block = dim3(tile);
  dim3 grid = dim3((m+tile-1)/tile, (n+tile-1)/tile);
  for (int i = 0; i < g_timing_iterations+2; i++) {
    if (i == 2) timer.start();
    kernel<<< grid, block >>>(m,
			      n,
			      k,
			      A,
			      B,
			      C2);
  }
  timer.stop();
  double tcutlass = timer.elapsed_millis() / g_timing_iterations;
  double cutlass_flops = double(num_flops) / tcutlass / 1.0e6;
  printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, cutlass_flops);
  double err = 0;
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      err += fabs(C[m*i+j] - C2[m*i+j]);
    }
  }
  printf("error: %lf\n", err/n/m);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(C2);
  cublasDestroy(g_cublas_handle);
}
