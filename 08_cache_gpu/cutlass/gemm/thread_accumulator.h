#pragma once

#include "../util/util.h"

namespace cutlass {
namespace gemm {

template <
    int ItemsPerThreadsY,
    int ItemsPerThreadsX>
struct thread_accumulator {

protected:
  float accumulators[ItemsPerThreadsY][ItemsPerThreadsX];

  inline __device__
    static void mad(
		    float &d,
		    const float &a,
		    const float &b,
		    const float &c)
    {
      asm volatile ( "fma.rn.f32 %0, %1, %2, %3;\n"
		     : "=f"(d) : "f"(a), "f"(b), "f"(c));
    }

  inline __device__
    void mad_xy(float (&tile_a)[ItemsPerThreadsY],
		float (&tile_b)[ItemsPerThreadsX],
		int x,
		int y) {
      mad(
	  accumulators[y][x],
	  tile_a[y],
	  tile_b[x],
	  accumulators[y][x]);
    }

public:
  inline __device__
    thread_accumulator() {}

  inline __device__ void init() {
#pragma unroll
    for (int y = 0; y < ItemsPerThreadsY; ++y) {
#pragma unroll
      for (int x = 0; x < ItemsPerThreadsX; ++x)
      {
	accumulators[y][x] = float(0);
      }
    }
  }

  inline __device__ float get(int x, int y) {
    return accumulators[y][x];
  }

  inline __device__
    void multiply_accumulate(
			     float (&tile_a)[ItemsPerThreadsY],
			     float (&tile_b)[ItemsPerThreadsX]) {
#pragma unroll
	for (int y = 0; y < ItemsPerThreadsY; ++y) {
#pragma unroll
	  for (int x = 0; x < ItemsPerThreadsX; ++x) {
	  mad_xy(tile_a, tile_b, x, y);
	}
      }
    }
};
}
}
