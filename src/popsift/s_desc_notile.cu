/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <stdio.h>
#include <iso646.h>

#include "sift_constants.h"
#include "s_gradiant.h"
#include "s_desc_notile.h"
#include "common/assist.h"
#include "common/vec_macros.h"

using namespace popsift;

__device__ static inline
void ext_desc_notile_sub( const float                  x,
                          const float                  y,
                          const int                    level,
                          const float                  cos_t,
                          const float                  sin_t,
                          const float                  SBP,
                          const Extremum* __restrict__ ext,
                          float* __restrict__          features,
                          cudaTextureObject_t          texLinear )
{
    const int ix   = threadIdx.y;
    const int iy   = threadIdx.z;
    const int tile = ( iy << 2 ) + ix;

    __shared__ float dpt[128];
    if( threadIdx.z < 2 ) {
        dpt[threadIdx.z * 64 + threadIdx.y * 16 + threadIdx.x] = 0.0f;
    }
    __syncthreads();

    const float csbp  = cos_t * SBP;
    const float ssbp  = sin_t * SBP;

    const float2 offset = make_float2( ix - 1.5f, iy - 1.5f );

    const float2 pt = make_float2( ::fmaf( csbp, offset.x, ::fmaf( -ssbp, offset.y, x ) ),
                                   ::fmaf( csbp, offset.y, ::fmaf(  ssbp, offset.x, y ) ) );

    // float dpt[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    const float2 lft_dn = make_float2( -cos_t + sin_t, -cos_t - sin_t );
    const float2 rgt_stp = make_float2(  cos_t, sin_t ) / 8.0f;
    const float2 up__stp = make_float2( -sin_t, cos_t ) / 8.0f;

    int xd = threadIdx.x;
    for( int yd=0; yd<16; yd++ )
    {
        float2 pixo = lft_dn + (xd+0.5f) * rgt_stp + (yd+0.5f) * up__stp;
        float2 pix  = pixo * SBP;

        float mod;
        float th;
        get_gradiant( mod, th, (pt+pix).x, (pt+pix).y, cos_t, sin_t, texLinear, level );
        th += ( th <  0.0f  ? M_PI2 : 0.0f ); //  if (th <  0.0f ) th += M_PI2;
        th -= ( th >= M_PI2 ? M_PI2 : 0.0f ); //  if (th >= M_PI2) th -= M_PI2;

        const float ww = d_consts.desc_gauss[iy*8+yd][ix*8+xd];
        const float wx = d_consts.desc_tile[xd];
        const float wy = d_consts.desc_tile[yd];

        const float  wgt = ww * wx * wy * mod;

        const float tth  = __fmul_ru( th, M_4RPI ); // th * M_4RPI;
        const int   fo   = (int)floorf(tth);
        const float do0  = tth - fo;
        const float wgt1 = 1.0f - do0;
        const float wgt2 = do0;

        int fo0  =   fo       % 8;
        int fo1  = ( fo + 1 ) % 8;
        atomicAdd( &dpt[tile*8+fo0], wgt * wgt1 );
        atomicAdd( &dpt[tile*8+fo1], wgt * wgt2 );
        __syncthreads();
    }

    if( threadIdx.z < 2 ) {
        const int idx = threadIdx.z * 64 + threadIdx.y * 16 + threadIdx.x;
        features[idx] = dpt[idx];
    }
}

__global__
void ext_desc_notile( Extremum*           extrema,
                      Descriptor*         descs,
                      int*                feat_to_ext_map,
                      cudaTextureObject_t texLinear )
{
    const int   offset   = blockIdx.x;
    Descriptor* desc     = &descs[offset];
    const int   ext_idx  = feat_to_ext_map[offset];
    Extremum*   ext      = &extrema[ext_idx];
    const int   ext_base = ext->idx_ori;
    const int   ext_num  = offset - ext_base;
    const float ang      = ext->orientation[ext_num];

    if( ext->sigma == 0 ) return;
    const float SBP   = fabsf(DESC_MAGNIFY * ext->sigma);

    float cos_t;
    float sin_t;
    __sincosf( ang, &sin_t, &cos_t );

    ext_desc_notile_sub( ext->xpos, ext->ypos, ext->lpos,
                         cos_t, sin_t, SBP,
                         ext,
                         desc->features,
                         texLinear );
}

