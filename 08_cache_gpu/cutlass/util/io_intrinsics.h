#pragma once

#include <stdint.h>
#include <cuda_fp16.h>
#include "nv_std.h"
#include "math.h"

namespace cutlass {

template <typename value_t, int VectorItems, int AlignBytes> struct io_vector_base;
template <typename value_t, int VectorItems> struct __align__(1) io_vector_base<value_t, VectorItems, 1> { value_t buff[VectorItems]; };
template <typename value_t, int VectorItems> struct __align__(2) io_vector_base<value_t, VectorItems, 2> { value_t buff[VectorItems]; };
template <typename value_t, int VectorItems> struct __align__(4) io_vector_base<value_t, VectorItems, 4> { value_t buff[VectorItems]; };
template <typename value_t, int VectorItems> struct __align__(8) io_vector_base<value_t, VectorItems, 8> { value_t buff[VectorItems]; };
template <typename value_t, int VectorItems> struct __align__(16) io_vector_base<value_t, VectorItems, 16> { value_t buff[VectorItems]; };

template <
    typename value_t,                                                           ///< Component value type
    int MaxVectorItems,                                                         ///< Maximum allowable component values
    int MaxAlignBytes                                                           ///< Maximum allowable alignment
            = __NV_STD_MIN(16, MaxVectorItems * sizeof(value_t)),
    int AlignBytes                                                              ///< Actual alignment
            = __NV_STD_MIN(sizeof(value_t) * MaxVectorItems, MaxAlignBytes),
    int VectorItems                                                             ///< Actual number of component values
            = divide_assert<AlignBytes, sizeof(value_t)>::value,
    bool MustAlias                                                              ///< Whether we need to alias during loads/stores
            = (VectorItems > 4)>
struct io_vector;


/**
 * IO vector (specialization for VectorItems <= 4)
 */
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
    io_vector_base<value_t, _VectorItems, _AlignBytes>
{
    enum
    {
        VectorItems = _VectorItems,
        AlignBytes = _AlignBytes
    };

    static_assert(is_pow2<AlignBytes>::value, "I/O vector alignment must be a power-of-two.");
    static_assert((AlignBytes <= 16), "I/O vector alignment must <= 16B.");

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
    void store(io_vector *ptr) const
    {
        *ptr = *this;
    }

    inline __device__
    void store(value_t *ptr) const
    {
        *reinterpret_cast<io_vector*>(ptr) = *this;
    }
};

template <typename ptr_t>
inline __device__
void ldg_cg_internal(float (&dest)[1],
		     ptr_t ptr) {
  asm volatile ("ld.global.cg.f32 %0, [%1];\n"
		:
		"=f"(dest[0])
		:
		"l"(ptr));
}

template <typename ptr_t>
inline __device__
void stg_cg_internal(ptr_t ptr,
		     const float (&src)[1]) {
  asm volatile ("st.global.cg.f32 [%0], %1;\n"
		: :
		"l"(ptr),
		"f"(src[0]));
}


/******************************************************************************
 * I/O cast types
 ******************************************************************************/

/// Provides the type for which to reinterpret-cast a given vector
struct io_cast {
    typedef float type[1];
};

template <typename ptr_t, typename value_t>
inline __device__
void ldg_cg(value_t &dest, ptr_t d_in) {
  ldg_cg_internal(reinterpret_cast<typename io_cast::type &>(dest),
		  d_in);
}

template <typename ptr_t, typename value_t>
inline __device__
void stg_cg(ptr_t dest, const value_t &src) {
  stg_cg_internal(dest,
		  reinterpret_cast<const typename io_cast::type &>(src));
}

} // namespace cutlass

