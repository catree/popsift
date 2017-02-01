/*
 * Copyright 2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "s_desc_normalize.h"

using namespace popsift;
using namespace std;

__device__ static inline
void normalize_histogram_root_sift_sub( int offset, const float* features, int num_orientations )
{
    float4* ptr4 = (float4*)features;

    float4 descr;
    descr = ptr4[threadIdx.x];

    float sum = descr.x + descr.y + descr.z + descr.w;

    sum += __shfl_down( sum, 16 );
    sum += __shfl_down( sum,  8 );
    sum += __shfl_down( sum,  4 );
    sum += __shfl_down( sum,  2 );
    sum += __shfl_down( sum,  1 );

    sum = __shfl( sum,  0 );

#if 1
    float val;
    val = scalbnf( __fsqrt_rn( __fdividef( descr.x, sum ) ),
                   d_consts.norm_multi );
    descr.x = val;
    val = scalbnf( __fsqrt_rn( __fdividef( descr.y, sum ) ),
                   d_consts.norm_multi );
    descr.y = val;
    val = scalbnf( __fsqrt_rn( __fdividef( descr.z, sum ) ),
                   d_consts.norm_multi );
    descr.z = val;
    val = scalbnf( __fsqrt_rn( __fdividef( descr.w, sum ) ),
                   d_consts.norm_multi );
    descr.w = val;
#else
    float val;
    val = 512.0f * __fsqrt_rn( __fdividef( descr.x, sum ) );
    descr.x = val;
    val = 512.0f * __fsqrt_rn( __fdividef( descr.y, sum ) );
    descr.y = val;
    val = 512.0f * __fsqrt_rn( __fdividef( descr.z, sum ) );
    descr.z = val;
    val = 512.0f * __fsqrt_rn( __fdividef( descr.w, sum ) );
    descr.w = val;
#endif

    const bool ignoreme = ( offset >= num_orientations );

    if( not ignoreme ) {
        ptr4[threadIdx.x] = descr;
    }
}

__global__
void normalize_histogram_root_sift( Descriptor* descs, int num_orientations )
{
    // root sift normalization

    int offset = blockIdx.x * 32 + threadIdx.y;

    // all of these threads are useless
    if( blockIdx.x * 32 >= num_orientations ) return;

    offset = ( offset < num_orientations ) ? offset
                                           : num_orientations-1;
    Descriptor* desc = &descs[offset];

    normalize_histogram_root_sift_sub( offset, desc->features, num_orientations );
}

