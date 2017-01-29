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

__global__
void normalize_histogram_l2( Descriptor* descs, int num_orientations )
{
    // OpenCV normalization

    int offset = blockIdx.x * 32 + threadIdx.y;

    // all of these threads are useless
    if( blockIdx.x * 32 >= num_orientations ) return;

    bool ignoreme = ( offset >= num_orientations );

    offset = ( offset < num_orientations ) ? offset
                                           : num_orientations-1;
    Descriptor* desc = &descs[offset];

    float*  ptr1 = desc->features;
    float4* ptr4 = (float4*)ptr1;

    float4 descr;
    descr = ptr4[threadIdx.x];

#undef HAVE_NORMF
    // normf() is an elegant function: sqrt(sum_0^127{v^2})
    // It exists from CUDA 7.5 but the trouble with CUB on the GTX 980 Ti forces
    // us to with CUDA 7.0 right now

#ifdef HAVE_NORMF
    float norm;

    if( threadIdx.x == 0 ) {
        norm = normf( 128, ptr1 );
    }

    norm = __shfl( norm,  0 );

    descr.x = min( descr.x, 0.2f*norm );
    descr.y = min( descr.y, 0.2f*norm );
    descr.z = min( descr.z, 0.2f*norm );
    descr.w = min( descr.w, 0.2f*norm );

    if( threadIdx.x == 0 ) {
        norm = scalbnf( rnormf( 128, ptr1 ), d_consts.norm_multi );
    }
#else
    float norm;

    norm = descr.x * descr.x
         + descr.y * descr.y
         + descr.z * descr.z
         + descr.w * descr.w;
    norm += __shfl_down( norm, 16 );
    norm += __shfl_down( norm,  8 );
    norm += __shfl_down( norm,  4 );
    norm += __shfl_down( norm,  2 );
    norm += __shfl_down( norm,  1 );
    if( threadIdx.x == 0 ) {
        norm = __fsqrt_rn( norm );
    }
    norm = __shfl( norm,  0 );

    descr.x = min( descr.x, 0.2f*norm );
    descr.y = min( descr.y, 0.2f*norm );
    descr.z = min( descr.z, 0.2f*norm );
    descr.w = min( descr.w, 0.2f*norm );

    norm = descr.x * descr.x
         + descr.y * descr.y
         + descr.z * descr.z
         + descr.w * descr.w;
    norm += __shfl_down( norm, 16 );
    norm += __shfl_down( norm,  8 );
    norm += __shfl_down( norm,  4 );
    norm += __shfl_down( norm,  2 );
    norm += __shfl_down( norm,  1 );
    if( threadIdx.x == 0 ) {
        // norm = __fsqrt_rn( norm );
        // norm = __fdividef( 512.0f, norm );
        norm = __frsqrt_rn( norm ); // inverse square root
        norm = scalbnf( norm, d_consts.norm_multi );
    }
#endif
    norm = __shfl( norm,  0 );

    descr.x = descr.x * norm;
    descr.y = descr.y * norm;
    descr.z = descr.z * norm;
    descr.w = descr.w * norm;

    if( not ignoreme ) {
        ptr4[threadIdx.x] = descr;
    }
}

