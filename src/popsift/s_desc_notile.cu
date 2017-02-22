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
void ext_desc_get_grad( const float                  x,
                        const float                  y,
                        const int                    level,
                        cudaTextureObject_t          texLinear,
                        const float                  cos_t,
                        const float                  sin_t,
                        const float                  SBP,
                        const int                    offx,
                        const int                    offy,
                        float&                       mod,
                        float&                       th )
{
    const float mvx = -2.5f + offx/8.0f + 1.0f/16.0f;
    const float mvy = -2.5f + offy/8.0f + 1.0f/16.0f;
    const float ptx  = ( cos_t * mvx - sin_t * mvy ) * SBP;
    const float pty  = ( cos_t * mvy + sin_t * mvx ) * SBP;
    get_gradiant( mod, th, x + ptx, y + pty, cos_t, sin_t, texLinear, level );
    th += ( th <  0.0f  ? M_PI2 : 0.0f ); //  if (th <  0.0f ) th += M_PI2;
    th -= ( th >= M_PI2 ? M_PI2 : 0.0f ); //  if (th >= M_PI2) th -= M_PI2;
}

__device__ static inline
void ext_desc_inc_tile( float* dpt, const int ix, const int iy, const int xd, const int yd, const float th, const float mod, const float ww )
{
    if( ix < 0 || iy < 0 || ix > 3 || iy > 3 ) return;

    const float wx = d_consts.desc_tile[xd];
    const float wy = d_consts.desc_tile[yd];

    const float  wgt = ww * wx * wy * mod;

    const float tth  = th * M_4RPI;
    const int   fo   = (int)floorf(tth);
    const float do0  = tth - fo;
    const float wgt1 = 1.0f - do0;
    const float wgt2 = do0;

    const int fo0  =   fo       & (8-1); // % 8;
    const int fo1  = ( fo + 1 ) & (8-1); // % 8;
    dpt[fo0] += ( wgt * wgt1 );
    dpt[fo1] += ( wgt * wgt2 );
}

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
    float dpt[2][32];

    {
        int iy = 0;
        memset( dpt[0], 0, 32*sizeof(float) );
        for( int ix=0; ix<5; ix++ ) {
            for( int yd = threadIdx.x / 8; yd < 8; yd += 4 ) {
                const int xd = threadIdx.x & (8-1);
                const int offx = ix*8+xd;
                const int offy = iy*8+yd;
                float mod, th;
                ext_desc_get_grad( x, y, level, texLinear, cos_t, sin_t, SBP, offx, offy, mod, th );
                float ww = d_consts.desc_gauss[offy][offx];
                int tile;

                tile = ix-1;
                ext_desc_inc_tile( &dpt[0][tile*8], ix-1, iy,   xd+8, yd,   th, mod, ww );

                tile = ix;
                ext_desc_inc_tile( &dpt[0][tile*8], ix,   iy,   xd,   yd,   th, mod, ww );
            }
        }
    }
    for( int iy=1; iy<5; iy++ ) {
        memset( dpt[iy&1?1:0], 0, 32*sizeof(float) );
        for( int ix=0; ix<5; ix++ ) {
            for( int yd = threadIdx.x / 8; yd < 8; yd += 4 ) {
                const int xd = threadIdx.x & (8-1);
                const int offx = ix*8+xd;
                const int offy = iy*8+yd;
                float mod, th;
                ext_desc_get_grad( x, y, level, texLinear, cos_t, sin_t, SBP, offx, offy, mod, th );
                float ww = d_consts.desc_gauss[offy][offx];
                int tile;

                tile = ix-1;
                ext_desc_inc_tile( &dpt[iy&1?0:1][tile*8], ix-1, iy-1, xd+8, yd+8, th, mod, ww );

                tile = ix;
                ext_desc_inc_tile( &dpt[iy&1?0:1][tile*8], ix,   iy-1, xd,   yd+8, th, mod, ww );

                tile = ix-1;
                ext_desc_inc_tile( &dpt[iy&1?1:0][tile*8], ix-1, iy,   xd+8, yd,   th, mod, ww );

                tile = ix;
                ext_desc_inc_tile( &dpt[iy&1?1:0][tile*8], ix,   iy,   xd,   yd,   th, mod, ww );
            }
        }

        __syncthreads();

        for( int i=0; i<32; i++ ) {
            float d = dpt[iy&1?0:1][i];
            d += __shfl_xor( d,  1 );
            d += __shfl_xor( d,  2 );
            d += __shfl_xor( d,  4 );
            d += __shfl_xor( d,  8 );
            d += __shfl_xor( d, 16 );
            dpt[iy&1?0:1][i] = d;
        }

        __syncthreads();

        features[(iy-1)*32+threadIdx.x] = dpt[iy&1?0:1][threadIdx.x];
    }
}

__global__
void ext_desc_notile( Extremum*           extrema,
                      const int           num,
                      Descriptor*         descs,
                      int*                feat_to_ext_map,
                      cudaTextureObject_t texLinear )
{
    const int   offset   = blockIdx.x * blockDim.z + threadIdx.z;
    if( offset >= num ) return;

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

