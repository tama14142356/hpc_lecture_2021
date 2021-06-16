#pragma once

#include <stdint.h>

namespace cutlass {

struct __align__(16) io_vector { float buff[4]; };

inline __device__
void stg_cg(float *ptr, const float &src) {
  asm volatile ("st.global.cg.f32 [%0], %1;\n"
		: :
		"l"(ptr),
		"f"(src));
}

} // namespace cutlass

