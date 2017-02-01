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
#include "s_desc_grid.h"
#include "assist.h"
#include "common/vec_macros.h"

using namespace popsift;

__device__ static inline
void ext_desc_grid( const float         ang,
                    const Extremum*     ext,
                    float* __restrict__ features,
                    Plane2D_float       layer,
                    cudaTextureObject_t layer_tex )
{
    const int width  = layer.getWidth();
    const int height = layer.getHeight();

    const int ix   = threadIdx.y;
    const int iy   = threadIdx.z;
    const int tile = ( ( ( iy << 2 ) + ix ) << 3 ); // base of the 8 floats written by this group of 16 threads

    const float x    = ext->xpos;
    const float y    = ext->ypos;
    const float sig  = ext->sigma;
    const float SBP  = fabsf(DESC_MAGNIFY * sig);

    if( SBP == 0 ) {
        return;
    }

    // const float cos_t = cosf(ang);
    // const float sin_t = sinf(ang);
    float cos_t;
    float sin_t;
    __sincosf( ang, &sin_t, &cos_t );

    const float csbp  = cos_t * SBP;
    const float ssbp  = sin_t * SBP;
    // const float crsbp = cos_t / SBP;
    // const float srsbp = sin_t / SBP;

    // const float offsetptx = ix - 1.5f;
    // const float offsetpty = iy - 1.5f;
    const float2 offset = make_float2( ix - 1.5f, iy - 1.5f );

    // The following 2 lines were the primary bottleneck of this kernel
    // const float ptx = csbp * offsetptx - ssbp * offsetpty + x;
    // const float pty = csbp * offsetpty + ssbp * offsetptx + y;
    // const float ptx = ::fmaf( csbp, offsetptx, ::fmaf( -ssbp, offsetpty, x ) );
    // const float pty = ::fmaf( csbp, offsetpty, ::fmaf(  ssbp, offsetptx, y ) );
    const float2 pt = make_float2( ::fmaf( csbp, offset.x, ::fmaf( -ssbp, offset.y, x ) ),
                                   ::fmaf( csbp, offset.y, ::fmaf(  ssbp, offset.x, y ) ) );

    /* At this point, we have the 16 centers (ptx,pty) of the 16 sections
     * of the SIFT descriptor.  */

    // const float bsz = fabsf(csbp) + fabsf(ssbp);
    // const int   xmin = max(1,          (int)floorf(ptx - bsz));
    // const int   ymin = max(1,          (int)floorf(pty - bsz));
    // const int   xmax = min(width - 2,  (int)floorf(ptx + bsz));
    // const int   ymax = min(height - 2, (int)floorf(pty + bsz));

    /* At this point, we have upright (unrotated) squares around the 16
     * points. These are meant o limit the search for pixels that are actually
     * inside the rotated square.
     * If we assume that sampling around in the rotated box is sufficient,
     * we uniformly sample points and let CUDA texture access solve the
     * location of the actual pixel by nearest neighbour search.
     * We could also try the linear interpolation method, hoping that
     * get_gradiant still returns feasible values. Note that these 2 ideas
     * should both be tested.
     */

    // const int wx = xmax - xmin + 1;
    // const int hy = ymax - ymin + 1;
    // const int loops = wx * hy;

    float dpt[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    // const float2 rgt_up = make_float2(  cos_t - sin_t,  cos_t + sin_t );
    // const float2 lft_up = make_float2( -cos_t - sin_t,  cos_t - sin_t );
    // const float2 rgt_dn = make_float2(  cos_t + sin_t, -cos_t + sin_t );
    const float2 lft_dn = make_float2( -cos_t + sin_t, -cos_t - sin_t );
    // const float2 rgt_stp = ( rgt_dn - lft_dn ) / 16.0f;
    // const float2 up__stp = ( lft_up - lft_dn ) / 16.0f;
    const float2 rgt_stp = make_float2(  cos_t, sin_t ) / 8.0f;
    const float2 up__stp = make_float2( -sin_t, cos_t ) / 8.0f;

    if( int(x)==177 && int(y)==591 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 )
    {
        printf("Found pixel (%d,%d)\n", int(x), int(y));
    }

    int xd = threadIdx.x;
    for( int yd=0; yd<16; yd++ )
    // for( int i = threadIdx.x; i < loops; i+=blockDim.x )
    {
        float2 pixo = lft_dn + (xd+0.5f) * rgt_stp + (yd+0.5f) * up__stp;
        float2 pix  = pixo * SBP;
        pix = round( pt + pix ) - pt;
        pixo = pix / SBP;

        float mod;
        float th;
        get_gradiant( mod, th, int((pt+pix).x), int((pt+pix).y), layer_tex );

        const float2 norm_pix = make_float2( ::fmaf( cos_t, pixo.x,  sin_t * pixo.y ),
                                             ::fmaf( cos_t, pixo.y, -sin_t * pixo.x ) );

        const float2 dn  = norm_pix + offset;
        const float  ww  = expf( -scalbnf(dn.x*dn.x + dn.y*dn.y, -3)); // expf(-0.125f * (dnx*dnx + dny*dny));
        const float2 w   = make_float2( 1.0f - fabsf(norm_pix.x),
                                        1.0f - fabsf(norm_pix.y) );

        if( w.x < 0.0f || w.y < 0.0f ) continue;

        const float  wgt = ww * w.x * w.y * mod;
    if( int(x)==177 && int(y)==591 )
    {
        int jj = int((pt+pix).x);
        int ii = int((pt+pix).y);
        printf("center pixel (%.2f,%.2f) ang %.2f check pixel (%d,%d) mod %.2f th %.2f ww %.2f wgt %.2f (GRID)\n", x, y, ang, jj, ii, mod, th, ww, wgt );
    }

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
        // dpt[i] += __shfl_down( dpt[i], 16 );
        dpt[i] += __shfl_down( dpt[i], 8, 16 );
        dpt[i] += __shfl_down( dpt[i], 4, 16 );
        dpt[i] += __shfl_down( dpt[i], 2, 16 );
        dpt[i] += __shfl_down( dpt[i], 1, 16 );
        dpt[i]  = __shfl     ( dpt[i], 0, 16 );
    }


    if( threadIdx.x < 8 ) {
        features[tile+threadIdx.x] = dpt[threadIdx.x];
    }
}

__global__
void ext_desc_grid( Extremum*           extrema,
                    Descriptor*         descs,
                    int*                feat_to_ext_map,
                    Plane2D_float       layer,
                    cudaTextureObject_t layer_tex )
{
    const int   offset   = blockIdx.x;
    Descriptor* desc     = &descs[offset];
    const int   ext_idx  = feat_to_ext_map[offset];
    Extremum*   ext      = &extrema[ext_idx];
    const int   ext_base = ext->idx_ori;
    const int   ext_num  = offset - ext_base;
    const float ang      = ext->orientation[ext_num];

    ext_desc_grid( ang,
                   ext,
                   desc->features,
                   layer,
                   layer_tex );
}

