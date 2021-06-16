#pragma once
#include <cuda_runtime.h>

namespace cutlass {

  struct gpu_timer {
    cudaEvent_t _start;
    cudaEvent_t _stop;

    gpu_timer() {
      cudaEventCreate(&_start);
      cudaEventCreate(&_stop);
    }

    ~gpu_timer() {
      cudaEventDestroy(_start);
      cudaEventDestroy(_stop);
    }

    void start() {
      cudaEventRecord(_start, 0);
    }

    void stop() {
      cudaEventRecord(_stop, 0);
    }

    float elapsed_millis() {
      float elapsed = 0.0;
      cudaEventSynchronize(_stop);
      cudaEventElapsedTime(&elapsed, _start, _stop);
      return elapsed;
    }
  };
} // namespace cutlass
