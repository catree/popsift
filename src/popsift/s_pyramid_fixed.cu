/*
 * Copyright 2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "sift_pyramid.h"
#include "sift_constants.h"
#include "gauss_filter.h"
#include "common/debug_macros.h"
#include "assist.h"
#include "common/clamp.h"

#include <iostream>
#include <stdio.h>

namespace popsift {

namespace gauss {

namespace fixedSpan {

namespace absoluteTexAddress {
/* read from point-addressable texture of image from previous octave */

template<int SHIFT, int WIDTH, int HEIGHT, int LEVELS>
__global__
void octave_fixed( cudaTextureObject_t src_data,
                   Plane2D_float       dst_data,
                   cudaSurfaceObject_t dog_data )
{
    const int IDx   = threadIdx.x;
    const int IDy   = threadIdx.y;
    const int IDz   = threadIdx.z;
    const int SPAN  = SHIFT + 1;
    const int w     = dst_data.getWidth();
    const int h     = dst_data.getHeight();
    const int level = IDz + 1;
    const int plane_rows = IDz * h;

    const float* filter = &d_gauss.abs_oN.filter[level*GAUSS_ALIGN];

    Plane2D_float destination( w, h,
                               dst_data.ptr( plane_rows ),
                               dst_data.getPitch() );

    const int idx = blockIdx.x * WIDTH      + IDx;
    const int idy = blockIdx.y * blockDim.y + IDy;

    const float TSHIFT = 0.5f;
    float       val    = tex2D<float>( src_data, idx-SHIFT+TSHIFT, idy+TSHIFT );

    float       fval   = val * filter[0];
    #pragma unroll
    for( int i=1; i<SPAN; i++ ) {
        val   = tex2D<float>( src_data, idx-SHIFT+TSHIFT, idy-i+TSHIFT )
              + tex2D<float>( src_data, idx-SHIFT+TSHIFT, idy+i+TSHIFT );
        fval += val * filter[i];
    }

    float out = fval * filter[0];
    #pragma unroll
    for( int i=1; i<SPAN; i++ ) {
        val  = __shfl_up( fval, i ) + __shfl_down( fval, i );
        out += val * filter[i];
    }
    val = __shfl_down( out, SHIFT );

    __shared__ float lx_val[HEIGHT][WIDTH][LEVELS];

    if( IDx < WIDTH ) {
        lx_val[IDy][IDx][IDz] = val;
    }
    __syncthreads();

    if( IDx < WIDTH ) {
        const float l0_val = tex2D<float>( src_data, idx+TSHIFT, idy+TSHIFT );
        const float dogval = ( IDz == 0 )
                           ? val - l0_val
                           : val - lx_val[IDy][IDx][IDz-1];

        const bool i_write = ( idx < w && idy < h );

        if( i_write ) {
            destination.ptr(idy)[idx] = val;

            surf2DLayeredwrite( dogval, dog_data,
                                idx*4, idy,
                                threadIdx.z,
                                cudaBoundaryModeZero );
        }
    }
}

} // namespace absoluteTexAddress

namespace relativeTexAddress {
/* read from ratio-addressable texture of input image */

template<int SHIFT, int WIDTH, int HEIGHT, int LEVELS>
__global__
void octave_fixed( cudaTextureObject_t src_data,
                   Plane2D_float       dst_data,
                   cudaSurfaceObject_t dog_data,
                   const float         tshift )
{
    const int IDx   = threadIdx.x;
    const int IDy   = threadIdx.y;
    const int IDz   = threadIdx.z;
    const int SPAN  = SHIFT + 1;
    const int w     = dst_data.getWidth();
    const int h     = dst_data.getHeight();
    const int level = IDz;
    const int plane_rows = IDz * h;

    const float* filter = &d_gauss.abs_o0.filter[level*GAUSS_ALIGN];

    Plane2D_float destination( w, h,
                               dst_data.ptr( plane_rows ),
                               dst_data.getPitch() );

    const int idx = blockIdx.x * WIDTH      + IDx;
    const int idy = blockIdx.y * blockDim.y + IDy;

    const float dst_w  = w;
    const float dst_h  = h;
    const float r_x_ko = ( idx-SHIFT+tshift ) / dst_w;

    /* This thread reads from cell IDx - SHIFT */
    float       val    = tex2D<float>( src_data,
                                       r_x_ko,
                                       ( idy+tshift ) / dst_h );

    /* Filter in Y-direction first */
    float       fval   = val * filter[0];
    #pragma unroll
    for( int i=1; i<SPAN; i++ ) {
        val   = tex2D<float>( src_data,
                              r_x_ko,
                              ( idy-i+tshift ) / dst_h )
              + tex2D<float>( src_data,
                              r_x_ko,
                              ( idy+i+tshift ) / dst_h );
        fval += val * filter[i];
    }

    /* Filter in X-direction afterards */
    float out = fval * filter[0];
    #pragma unroll
    for( int i=1; i<SPAN; i++ ) {
        val  = __shfl_up( fval, i ) + __shfl_down( fval, i );
        out += val * filter[i];
    }
    val = __shfl_down( out, SHIFT+1 );

    val *= 255.0f; // don't forget to upscale

    __shared__ float lx_val[HEIGHT][WIDTH][LEVELS];

    if( IDx < WIDTH ) {
        lx_val[IDy][IDx][IDz] = val;
    }
    __syncthreads();

    if( IDx < WIDTH ) {

        const bool i_write = ( idx < w && idy < h );

        if( i_write ) {
            destination.ptr(idy)[idx] = val;

            if( IDz > 0 ) {
                float dogval = val - lx_val[IDy][IDx][IDz-1];
                if(IDx==1) dogval=0;
                // left side great
                // right side buggy
                surf2DLayeredwrite( dogval, dog_data,
                                    idx*4, idy,
                                    threadIdx.z-1,
                                    cudaBoundaryModeZero );
            }
        }
    }
}

} // namespace relativeTexAddress

} // namespace fixedSpan

} // namespace gauss

template<int SHIFT, bool OCT_0, int LEVELS>
__host__
inline void make_octave_sub( const Config& conf, Image* base, Octave& oct_obj, cudaStream_t stream )
{
    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    if( OCT_0 ) {
        const int x_size = 32;
        const int l_conf = LEVELS;
        const int w_conf = x_size - 2 * SHIFT;
        const int h_conf = 1024 / ( x_size * l_conf );
        dim3 block( x_size, h_conf, l_conf );
        dim3 grid;
        grid.x = grid_divide( width, w_conf );
        grid.y = grid_divide( height, block.y );

        assert( block.x * block.y * block.z < 1024 );

        const float tshift = 0.5f * powf( 2.0f, conf.getUpscaleFactor() );

        gauss::fixedSpan::relativeTexAddress::octave_fixed
            <SHIFT,w_conf,h_conf,l_conf>
            <<<grid,block,0,stream>>>
            ( base->getInputTexture(),
              oct_obj.getData(0),
              oct_obj.getDogSurface( ),
              tshift );
    } else {
        const int x_size = 32;
        const int l_conf = LEVELS-1;
        const int w_conf = x_size - 2 * SHIFT;
        const int h_conf = 1024 / ( x_size * l_conf );
        dim3 block( x_size, h_conf, l_conf );
        dim3 grid;
        grid.x = grid_divide( width, w_conf );
        grid.y = grid_divide( height, block.y );

        assert( block.x * block.y * block.z < 1024 );

        gauss::fixedSpan::absoluteTexAddress::octave_fixed
            <SHIFT,w_conf,h_conf,l_conf>
            <<<grid,block,0,stream>>>
            ( oct_obj._data_tex[0],
              oct_obj.getData(1),
              oct_obj.getDogSurface( ) );
    }
}

void Pyramid::make_octave( const Config& conf, Image* base, Octave& oct_obj, cudaStream_t stream, bool isOctaveZero )
{
    if( _levels == 6 ) {
        if( conf.getGaussMode() == Config::Fixed9 ) {
            if( isOctaveZero )
                make_octave_sub<4,true,6> ( conf, base, oct_obj, stream );
            else
                make_octave_sub<4,false,6>( conf, base, oct_obj, stream );
        } else if( conf.getGaussMode() == Config::Fixed15 ) {
            if( isOctaveZero )
                make_octave_sub<7,true,6> ( conf, base, oct_obj, stream );
            else
                make_octave_sub<7,false,6>( conf, base, oct_obj, stream );
        } else {
            POP_FATAL("Unsupported Gauss filter mode for making all octaves at once");
        }
    } else {
        POP_FATAL("Unsupported number of levels for making all octaves at once");
    }
}

} // namespace popsift

