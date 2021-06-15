#pragma once

#include "../util/util.h"
#include "dp_accummulate.h"

namespace cutlass {
namespace gemm {

template <
    int ThreadItemsY,
    int ThreadItemsX>
struct thread_accumulator {
protected:

  typedef dp_accummulate<float, float> dp_floatraits_t;

public:
  struct scratch_storage_t {};

protected:
  float accumulators[ThreadItemsY][ThreadItemsX];

  inline __device__
    void mad_xy(float (&tile_a)[ThreadItemsY],
		float (&tile_b)[ThreadItemsX],
		int x,
		int y) {
      dp_floatraits_t::mad(
			     accumulators[y][x],
			     tile_a[y],
			     tile_b[x],
			     accumulators[y][x]);
    }

public:
  inline __device__
    thread_accumulator(scratch_storage_t &scratch) {}

  inline __device__ void init() {
#pragma unroll
    for (int y = 0; y < ThreadItemsY; ++y) {
#pragma unroll
      for (int x = 0; x < ThreadItemsX; ++x)
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
			     float (&tile_a)[ThreadItemsY],
			     float (&tile_b)[ThreadItemsX]) {
#pragma unroll
      for (int y = 0; y < ThreadItemsY; ++y) {
#pragma unroll
	for (int x = 0; x < ThreadItemsX; ++x) {
	  mad_xy(tile_a, tile_b, x, y);
	}
      }
    }
};
}
}
