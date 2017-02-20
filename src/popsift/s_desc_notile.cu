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

#if 0

__device__ static inline
void ext_desc_notile_sub( const float         x,
                          const float         y,
                          const int           level,
                          const float         SBP,
                          const float         cos_t,
                          const float         sin_t,
                          const Extremum*     ext,
                          float* __restrict__ features,
                          cudaTextureObject_t texLinear )
{
    const float csbp  = cos_t * SBP;
    const float ssbp  = sin_t * SBP;

    __shared__ float dpt[16][8];

#warning must init dpt

    const float2 lft_dn  = make_float2( -cos_t + sin_t, -cos_t - sin_t );
    const float rgt_stpx =  cos_t / 8.0f;
    const float rgt_stpy =  sin_t / 8.0f;
    const float up__stpx = -sin_t / 8.0f;
    const float up__stpy =  cos_t / 8.0f;

    for( int ix=0; ix<4; ix++ ) {
        for( int iy=0; iy<4; iy++ ) {
            const int tile = ( ( iy << 2 ) + ix );

            int xd = threadIdx.x;
            for( int yd=0; yd<16; yd++ )
            {
                float pixox = lft_dn.x + (xd+0.5f) * rgt_stpx + (yd+0.5f) * up__stpx;
                float pixoy = lft_dn.y + (xd+0.5f) * rgt_stpy + (yd+0.5f) * up__stpy;

                const float norm_pixx = cos_t * pixox + sin_t * pixoy;
                const float norm_pixy = cos_t * pixoy - sin_t * pixox;

                const float wx        = 1.0f - fabsf(norm_pixx);
                const float wy        = 1.0f - fabsf(norm_pixy);

                const float offsetx = ( ix - 1.5f );
                const float offsety = ( iy - 1.5f );
                float ptx = pixox + cos_t * offsetx + (-sin_t) * offsety;
                float pty = pixoy + cos_t * offsety +   sin_t  * offsetx;
                ptx *= SBP;
                pty *= SBP;

                float mod;
                float th;
                get_gradiant( mod, th, x+ptx, y+pty, cos_t, sin_t, texLinear, level );
                th += ( th <  0.0f  ? M_PI2 : 0.0f ); //  if (th <  0.0f ) th += M_PI2;
                th -= ( th >= M_PI2 ? M_PI2 : 0.0f ); //  if (th >= M_PI2) th -= M_PI2;

                const float dnx  = norm_pixx + offsetx;
                const float dny  = norm_pixy + offsety;
                const float  ww  = expf( -scalbnf(dnx*dnx + dny*dny, -3)); // expf(-0.125f * (dnx*dnx + dny*dny));
// Note: it seems possible to precompute ww by simply walking through the 40x40 matrix of pixels
//       that I am extracting. The combination of offset and norm_pix seems seems to yield
//       weights that are in normalized space around (x,y).
//       It should be possible to pre-compute a __constant__ 40x40 matrix of weights!

                if( w.x < 0.0f || w.y < 0.0f ) continue;

                const float  wgt = ww * wx * wy * mod;

                const float tth  = __fmul_ru( th, M_4RPI ); // th * M_4RPI;
                const int   fo   = (int)floorf(tth);
                const float do0  = tth - fo0;             
                const float wgt1 = 1.0f - do0;
                const float wgt2 = do0;

                int fo0 = fo       % DESC_BINS;
                int fo1 = ( fo+1 ) % DESC_BINS;
                atomicAdd( &dpt[tile][fo0], wgt1 * wgt );
                atomicAdd( &dpt[tile][fo1], wgt2 * wgt );
            }
            __syncthreads();

            if( threadIdx.x < 8 ) {
                features[ (tile<<3) +threadIdx.x] = dpt[threadIdx.x];
            }
        }
    }
}

#endif

__global__
void ext_desc_notile( popsift::Extremum*     extrema,
                      popsift::Descriptor*   descs,
                      int*                   feat_to_ext_map,
                      cudaTextureObject_t    texLinear )
{
#if 0
    const int   offset   = blockIdx.x;
    Descriptor* desc     = &descs[offset];
    const int   ext_idx  = feat_to_ext_map[offset];
    Extremum*   ext      = &extrema[ext_idx];
    if( sig->sigma == 0 ) return;

    const int   ext_base = ext->idx_ori;
    const int   ext_num  = offset - ext_base;
    const float ang      = ext->orientation[ext_num];


    float cos_t;
    float sin_t;
    __sincosf( ang, &sin_t, &cos_t );

    ext_desc_notile_sub( ext->xpos,
                         ext->ypos,
                         ext->lpos,
                         fabsf(DESC_MAGNIFY * ext->sigma),
                         cos_t,
                         sin_t,
                         ext,
                         desc->features,
                         texLinear );
#endif
}

