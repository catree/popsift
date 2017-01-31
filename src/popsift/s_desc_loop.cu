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
#include "s_desc_loop.h"
#include "assist.h"
#include "common/vec_macros.h"

using namespace popsift;

__device__ static inline
void ext_desc_loop( const float         ang,
                    const Extremum*     ext,
                    float* __restrict__ features,
                    Plane2D_float       layer,
                    cudaTextureObject_t layer_tex )
{
    const int width  = layer.getWidth();
    const int height = layer.getHeight();

    // int bidx = blockIdx.x & 0xf; // lower 4 bits of block ID
    const int ix   = threadIdx.y; // bidx & 0x3;       // lower 2 bits of block ID
    const int iy   = threadIdx.z; // bidx >> 2;        // next lowest 2 bits of block ID

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
    const float crsbp = cos_t / SBP;
    const float srsbp = sin_t / SBP;

    const float2 offsetpt = make_float2( ix - 1.5f,
                                         iy - 1.5f );

    // The following 2 lines were the primary bottleneck of this kernel
    // const float ptx = csbp * offsetptx - ssbp * offsetpty + x;
    // const float pty = csbp * offsetpty + ssbp * offsetptx + y;
    const float ptx = ::fmaf( csbp, offsetpt.x, ::fmaf( -ssbp, offsetpt.y, x ) );
    const float pty = ::fmaf( csbp, offsetpt.y, ::fmaf(  ssbp, offsetpt.x, y ) );

    /* At this point, we have the 16 centers (ptx,pty) of the 16 sections
     * of the SIFT descriptor.  */

    const float bsz = fabsf(csbp) + fabsf(ssbp);
    const int   xmin = max(1,          (int)floorf(ptx - bsz));
    const int   ymin = max(1,          (int)floorf(pty - bsz));
    const int   xmax = min(width - 2,  (int)floorf(ptx + bsz));
    const int   ymax = min(height - 2, (int)floorf(pty + bsz));

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

    const int wx = xmax - xmin + 1;
    const int hy = ymax - ymin + 1;
    const int loops = wx * hy;

    float dpt[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    if( int(x)==177 && int(y)==591 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 )
    {
        printf("Found pixel (%d,%d)\n", int(x), int(y));
    }

    for( int i = threadIdx.x; i < loops; i+=blockDim.x )
    {
        const int ii = i / wx + ymin;
        const int jj = i % wx + xmin;     

        const float2 d = make_float2( jj - ptx, ii - pty );

        // const float nx = crsbp * dx + srsbp * dy;
        // const float ny = crsbp * dy - srsbp * dx;
        const float2 n = make_float2( ::fmaf( crsbp, d.x,  srsbp * d.y ),
                                      ::fmaf( crsbp, d.y, -srsbp * d.x ) );
        const float2 nn = abs(n);
        if (nn.x < 1.0f && nn.y < 1.0f) {
            // const float2 mod_th = get_gradiant( jj, ii, layer );
            const float2 mod_th = get_gradiant( jj, ii, layer_tex );
            const float& mod    = mod_th.x;
            float        th     = mod_th.y;

            const float2 dn = n + offsetpt;
            const float  ww = __expf( -scalbnf(dn.x*dn.x + dn.y*dn.y, -3)); // speedup !
            // const float ww  = __expf(-0.125f * (dnx*dnx + dny*dny)); // speedup !
            const float2 w  = make_float2( 1.0f - nn.x,
                                           1.0f - nn.y );
            const float wgt = ww * w.x * w.y * mod;
    if( int(x)==177 && int(y)==591 )
    {
        printf("center pixel (%.2f,%.2f) ang %.2f check pixel (%d,%d) mod %.2f th %.2f ww %.2f wgt %.2f (LOOP)\n", x, y, ang, jj, ii, mod, th, ww, wgt );
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
            // if(fo < 8) {
                // maf: multiply-add
                // _ru - round to positive infinity equiv to froundf since always >=0
            dpt[fo]   = __fmaf_ru( wgt1, wgt, dpt[fo] );   // dpt[fo]   += (wgt1*wgt);
            dpt[fo+1] = __fmaf_ru( wgt2, wgt, dpt[fo+1] ); // dpt[fo+1] += (wgt2*wgt);
            // }
        }
        __syncthreads();
    }

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

    // int hid    = blockIdx.x % 16;
    // int offset = hid*8;
    int offset = ( ( ( threadIdx.z << 2 ) + threadIdx.y ) << 3 ); // ( ( threadIdx.z * 4 ) + threadIdx.y ) * 8;

    if( threadIdx.x < 8 ) {
        features[offset+threadIdx.x] = dpt[threadIdx.x];
    }
}

__global__
void ext_desc_loop( Extremum*           extrema,
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

    ext_desc_loop( ang,
                   ext,
                   desc->features,
                   layer,
                   layer_tex );
}

