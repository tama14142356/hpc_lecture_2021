#pragma once
#include "../util/util.h"

namespace cutlass {
namespace gemm {

struct load_algorithm
{
    enum kind_t
    {
        CongruousCopy  = 0,
        CrosswiseCopy  = 1,
    };

    template <kind_t Kind>
    struct any_tag : nv_std::integral_constant<kind_t, Kind> {};

    typedef any_tag<CongruousCopy> contiguous_tag_t;

    typedef any_tag<CrosswiseCopy> crosswise_tag_t;

};


template <
    int BlockThreads,                       ///< Number of threads in each thread block (blockDim.x)
    int BlockDpVectorsK,                    ///< Extent of block-wide tile in float along the K-axis (height)
    int BlockDpVectorsL,                    ///< Extent of block-wide tile in float along the L-axis (width)
    int LeadingDimAlignBytes,               ///< Byte alignment of input matrix leading dimension
    load_algorithm::kind_t LoadAlgorithm>   ///< Algorithm for loading a shared tile of KxL matrix data
struct block_loader
{
    //-------------------------------------------------------------------------
    // Constructor API
    //-------------------------------------------------------------------------

    /// Constructor
    block_loader(
        float *d_matrix,              ///< Pointer to input matrix
        int matrix_values_l,            ///< Extent of the input matrix in float along the L-axis
        int matrix_values_stride_k,     ///< Distance in float within pitched-linear memory between successive coordinates along the K-axis
        int matrix_values_stride_l,     ///< Distance in float within pitched-linear memory between successive coordinates along the L-axis
        int2 block_begin_item_coords,   ///< Thread block's starting float coordinates (l, k) within the input matrix
        int block_end_item_k);          ///< Thread block's ending coordinate (k) within the input matrix (one-past)

    void request();

    void next();

    template <int _BlockDpVectorsL>
    void commit(
        float (&scratch_tile)[BlockDpVectorsK][_BlockDpVectorsL]);

};


} // namespace gemm
} // namespace cutlass

#include "block_loader_crosswise.h"
#include "block_loader_congruous_dp1.h"
