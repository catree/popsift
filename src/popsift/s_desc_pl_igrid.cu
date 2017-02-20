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
#include "s_desc_pl_igrid.h"
#include "common/assist.h"
#include "common/vec_macros.h"

using namespace popsift;

__device__ static inline
void ext_desc_pl_load_igrid( float2              gradcache[40][40],
                             const float         ang,
                             const float         sig,
                             const Extremum*     ext,
                             cudaTextureObject_t layer_tex )
{
    const float x     = ext->xpos;
    const float y     = ext->ypos;
    const int   level = ext->lpos; // old_level;

    const float SBP  = fabsf(DESC_MAGNIFY * sig);

    float cos_t;
    float sin_t;
    __sincosf( ang, &sin_t, &cos_t );

    const float csbp  = cos_t * SBP;
    const float ssbp  = sin_t * SBP;

    const float2 offset = make_float2( - 1.5f, - 1.5f );

    const float2 pt = make_float2( ::fmaf( csbp, offset.x, ::fmaf( -ssbp, offset.y, x ) ),
                                   ::fmaf( csbp, offset.y, ::fmaf(  ssbp, offset.x, y ) ) );

    const float2 lft_dn  = make_float2( -cos_t + sin_t, -cos_t - sin_t );
    const float2 rgt_stp = make_float2(  cos_t, sin_t ) / 8.0f;
    const float2 up__stp = make_float2( -sin_t, cos_t ) / 8.0f;

    for( int yd = threadIdx.z*4+threadIdx.y; yd<40; yd += 16 )
    {
        for( int xd = threadIdx.x; ::__any(xd<40); xd += 32 ) if( xd<40 )
        {
            float2 pixo = lft_dn + (xd+0.5f) * rgt_stp + (yd+0.5f) * up__stp;
            float2 pix  = pixo * SBP;
            // pix = round( pt + pix ) - pt;
            // pixo = pix / SBP;

            get_gradiant( gradcache[yd][xd].x,
                          gradcache[yd][xd].y,
                          (pt+pix).x,
                          (pt+pix).y,
                          layer_tex,
                          level );
        }

        ::__any(yd<40); // fake a barrier for a single warp only
    }
    __syncthreads();
}

__device__ static inline
void ext_desc_pl_igrid_sub( const int           ix,
                            const int           iy,
                            const float         ang,
                            const Extremum*     ext,
                            float* __restrict__ features,
                            const float2        gradcache[40][40] )
{
    const int tile = ( ( ( iy << 2 ) + ix ) << 3 ); // base of the 8 floats written by this group of 16 threads

    const float x    = ext->xpos;
    const float y    = ext->ypos;
    const float sig  = ext->sigma;
    const float SBP  = fabsf(DESC_MAGNIFY * sig);

    float cos_t;
    float sin_t;
    __sincosf( ang, &sin_t, &cos_t );

    const float csbp  = cos_t * SBP;
    const float ssbp  = sin_t * SBP;

    const float2 offset = make_float2( ix - 1.5f, iy - 1.5f );

    // The following 2 lines were the primary bottleneck of this kernel
    // const float ptx = csbp * offsetptx - ssbp * offsetpty + x;
    // const float pty = csbp * offsetpty + ssbp * offsetptx + y;
    // const float ptx = ::fmaf( csbp, offsetptx, ::fmaf( -ssbp, offsetpty, x ) );
    // const float pty = ::fmaf( csbp, offsetpty, ::fmaf(  ssbp, offsetptx, y ) );
    const float2 pt = make_float2( ::fmaf( csbp, offset.x, ::fmaf( -ssbp, offset.y, x ) ),
                                   ::fmaf( csbp, offset.y, ::fmaf(  ssbp, offset.x, y ) ) );

    float dpt[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    // const float2 rgt_up = make_float2(  cos_t - sin_t,  cos_t + sin_t );
    // const float2 lft_up = make_float2( -cos_t - sin_t,  cos_t - sin_t );
    // const float2 rgt_dn = make_float2(  cos_t + sin_t, -cos_t + sin_t );
    const float2 lft_dn = make_float2( -cos_t + sin_t, -cos_t - sin_t );
    // const float2 rgt_stp = ( rgt_dn - lft_dn ) / 16.0f;
    // const float2 up__stp = ( lft_up - lft_dn ) / 16.0f;
    const float2 rgt_stp = make_float2(  cos_t, sin_t ) / 8.0f;
    const float2 up__stp = make_float2( -sin_t, cos_t ) / 8.0f;

    int xd = threadIdx.x & 0xf;
    int yd = ( threadIdx.x >> 4 );
    for( ; yd<16; yd+=2 )
    {
        float2 pixo = lft_dn + (xd+0.5f) * rgt_stp + (yd+0.5f) * up__stp;
        // float2 pix  = pixo * SBP;
        // pix = round( pt + pix ) - pt;
        // pixo = pix / SBP;

        float mod = gradcache[iy*8+yd][ix*8+xd].x;
        float th  = gradcache[iy*8+yd][ix*8+xd].y;

        const float2 norm_pix = make_float2( ::fmaf( cos_t, pixo.x,  sin_t * pixo.y ),
                                             ::fmaf( cos_t, pixo.y, -sin_t * pixo.x ) );

        const float2 dn  = norm_pix + offset;
        const float  ww  = expf( -scalbnf(dn.x*dn.x + dn.y*dn.y, -3)); // expf(-0.125f * (dnx*dnx + dny*dny));
        const float2 w   = make_float2( 1.0f - fabsf(norm_pix.x),
                                        1.0f - fabsf(norm_pix.y) );

        if( w.x < 0.0f || w.y < 0.0f ) continue;

        const float  wgt = ww * w.x * w.y * mod;

        th -= ang;
        th += ( th <  0.0f  ? M_PI2 : 0.0f ); //  if (th <  0.0f ) th += M_PI2;
        th -= ( th >= M_PI2 ? M_PI2 : 0.0f ); //  if (th >= M_PI2) th -= M_PI2;

        const float tth  = __fmul_ru( th, M_4RPI ); // th * M_4RPI;
        const int   fo0  = (int)floorf(tth);
        const float do0  = tth - fo0;             
        const float wgt1 = 1.0f - do0;
        const float wgt2 = do0;

        int fo  = fo0 % DESC_BINS;
        dpt[fo]   = __fmaf_ru( wgt1, wgt, dpt[fo] );   // dpt[fo]   += (wgt1*wgt);
        dpt[fo+1] = __fmaf_ru( wgt2, wgt, dpt[fo+1] ); // dpt[fo+1] += (wgt2*wgt);
    }
    __syncthreads();

    dpt[0] += dpt[8];

    /* reduction here */
    for (int i = 0; i < 8; i++) {
        dpt[i] += __shfl_down( dpt[i], 16 );
        dpt[i] += __shfl_down( dpt[i], 8 );
        dpt[i] += __shfl_down( dpt[i], 4 );
        dpt[i] += __shfl_down( dpt[i], 2 );
        dpt[i] += __shfl_down( dpt[i], 1 );
        dpt[i]  = __shfl     ( dpt[i], 0 );
    }


    if( threadIdx.x < 8 ) {
        features[tile+threadIdx.x] = dpt[threadIdx.x];
    }
}

__global__
void ext_desc_pl_igrid( Extremum*           extrema,
                        Descriptor*         descs,
                        int*                feat_to_ext_map,
                        cudaTextureObject_t layer_tex )
{
    const int   ix       = threadIdx.y;
    const int   iy       = threadIdx.z;
    const int   offset   = blockIdx.x;

    __shared__ float2 gradcache[40][40];

    Descriptor* desc     = &descs[offset];
    const int   ext_idx  = feat_to_ext_map[offset];
    Extremum*   ext      = &extrema[ext_idx];
    const int   ext_base = ext->idx_ori;
    const int   ext_num  = offset - ext_base;
    const float ang      = ext->orientation[ext_num];
    const float sig      = ext->sigma;

    if( sig == 0 ) return;

    ext_desc_pl_load_igrid( gradcache,
                            ang,
                            sig,
                            ext,
                            layer_tex );

    ext_desc_pl_igrid_sub( ix,
                          iy,
                          ang,
                          ext,
                          desc->features,
                          gradcache );
}

