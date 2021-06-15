#pragma once

#include <stdint.h>

namespace cutlass {

template <typename value_t, int VectorItems, int AlignBytes> struct io_vector_base;
template <typename value_t, int VectorItems> struct __align__(16) io_vector_base<value_t, VectorItems, 16> { value_t buff[VectorItems]; };

template <
    typename value_t,
    int MaxVectorItems,
    int MaxAlignBytes = 16,
    int AlignBytes = MaxAlignBytes,
    int VectorItems = 4,
    bool MustAlias = false>
struct io_vector;

template <
    typename value_t,
    int MaxVectorItems,
    int MaxAlignBytes,
    int _AlignBytes,
    int _VectorItems>
struct io_vector <
    value_t,
    MaxVectorItems,
    MaxAlignBytes,
    _AlignBytes,
    _VectorItems,
    false>
:
    io_vector_base<value_t, _VectorItems, _AlignBytes> {
    inline __device__
    void load(const io_vector *ptr)
    {
        *this = *ptr;
    }
    inline __device__
    void load(const value_t *ptr)
    {
        *this = *reinterpret_cast<const io_vector*>(ptr);
    }
    inline __device__
    void store(value_t *ptr) const
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

