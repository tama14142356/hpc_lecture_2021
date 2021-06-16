#pragma once

#include <stdint.h>

namespace cutlass {

struct __align__(16) io_vector_base { float buff[4]; };

struct io_vector :
    io_vector_base {
    inline __device__
    void load(const io_vector *ptr)
    {
        *this = *ptr;
    }
    inline __device__
    void load(const float *ptr)
    {
        *this = *reinterpret_cast<const io_vector*>(ptr);
    }
    inline __device__
    void store(float *ptr) const
    {
        *reinterpret_cast<io_vector*>(ptr) = *this;
    }
};

inline __device__
void stg_cg(float *ptr, const float &src) {
  asm volatile ("st.global.cg.f32 [%0], %1;\n"
		: :
		"l"(ptr),
		"f"(src));
}

} // namespace cutlass

