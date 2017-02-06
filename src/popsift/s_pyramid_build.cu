/*
 * Copyright 2016, Simula Research Laboratory
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

/* It makes no sense whatsoever to change this value */
#define PREV_LEVEL 3

namespace popsift {

namespace gauss {

namespace variableSpan {

namespace absoluteTexAddress {
__global__
void horiz( cudaTextureObject_t src_data,
            Plane2D_float       dst_data,
            const int           dst_w,
            const int           dst_h,
            const int           dst_level )
{
    const int    src_level = dst_level - 1;
    const int    span      = d_gauss.inc.span[dst_level];
    const float* filter    = &popsift::d_gauss.inc.filter[dst_level*GAUSS_ALIGN];

    const int off_x = blockIdx.x * blockDim.x + threadIdx.x;

    if( off_x >= dst_w ) return;

    float out = 0.0f;

    #pragma unroll
    for( int offset = span; offset>0; offset-- ) {
        const float& g  = filter[offset];
        const float  v1 = tex2DLayered<float>( src_data, off_x - offset + 0.5f, blockIdx.y + 0.5f, src_level );
        out += ( v1 * g );

        const float  v2 = tex2DLayered<float>( src_data, off_x + offset + 0.5f, blockIdx.y + 0.5f, src_level );
        out += ( v2 * g );
    }
    const float& g  = filter[0];
    const float v3 = tex2D<float>( src_data, off_x+0.5f, blockIdx.y+0.5f );
    out += ( v3 * g );

    dst_data.ptr(blockIdx.y)[off_x] = out;
}

__device__ static inline
void vert_sub( cudaTextureObject_t src_data,
               cudaSurfaceObject_t dst_data,
               const int           dst_w,
               const int           dst_h,
               const int           dst_level,
               const int           span,
               const float*        filter )
{
    int block_x = blockIdx.x * blockDim.x;
    int block_y = blockIdx.y * blockDim.y;
    int idx     = threadIdx.x;
    int idy;

    float g;
    float val;
    float out = 0;

    for( int offset = span; offset>0; offset-- ) {
        g  = filter[offset];

        idy = threadIdx.y - offset;
        val = tex2D<float>( src_data, block_x + idx + 0.5f, block_y + idy + 0.5f );
        out += ( val * g );

        idy = threadIdx.y + offset;
        val = tex2D<float>( src_data, block_x + idx + 0.5f, block_y + idy + 0.5f );
        out += ( val * g );
    }

    g  = filter[0];
    idy = threadIdx.y;
    val = tex2D<float>( src_data, block_x + idx + 0.5f, block_y + idy + 0.5f );
    out += ( val * g );

    idx = block_x+threadIdx.x;
    idy = block_y+threadIdx.y;

    if( idx >= dst_w ) return;
    if( idy >= dst_h ) return;

    surf2DLayeredwrite( out, dst_data, idx*4, idy, dst_level, cudaBoundaryModeZero ); // dst_data.ptr(idy)[idx] = out;
}

__global__
void vert( cudaTextureObject_t src_data,
           cudaSurfaceObject_t dst_data,
           int                 dst_w,
           int                 dst_h,
           int                 dst_level )
{
    vert_sub( src_data, dst_data, dst_w, dst_h, dst_level, d_gauss.inc.span[dst_level], &popsift::d_gauss.inc.filter[dst_level*GAUSS_ALIGN] );
}

} // namespace absoluteTexAddress

namespace relativeTexAddress {

__device__
inline static void horiz_sub( cudaTextureObject_t src_data,
                              Plane2D_float       dst_data,
                              float               shift,
                              int                 span,
                              float*              filter )
{
    const float dst_w  = dst_data.getWidth();
    const float dst_h  = dst_data.getHeight();
    const float read_y = ( blockIdx.y + shift ) / dst_h;

    const int off_x = blockIdx.x * blockDim.x + threadIdx.x;

    if( off_x >= dst_w ) return;

    float out = 0.0f;

    #pragma unroll
    for( int offset = span; offset>0; offset-- ) {
        const float& g  = filter[offset];
        const float read_x_l = ( off_x - offset );
        const float  v1 = tex2D<float>( src_data, ( read_x_l + shift ) / dst_w, read_y );
        out += ( v1 * g );

        const float read_x_r = ( off_x + offset );
        const float  v2 = tex2D<float>( src_data, ( read_x_r + shift ) / dst_w, read_y );
        out += ( v2 * g );
    }
    const float& g  = filter[0];
    const float read_x = off_x;
    const float v3 = tex2D<float>( src_data, ( read_x + shift ) / dst_w, read_y );
    out += ( v3 * g );

    dst_data.ptr(blockIdx.y)[off_x] = out * 255.0f;
}

__global__
void horiz( cudaTextureObject_t src_data,
            Plane2D_float       dst_data,
            int                 octave,
            float               shift )
{
    // The first line creates level-0 octave-0 for the input image only.
    // Since we are computing the direct-downscaling gauss filter tables
    // and the first entry in that table is identical to the "normal"
    // table, we do not need a special case.
    // horiz( src_data, dst_data, shift, d_gauss.inc.span[0], &d_gauss.inc.filter[0*GAUSS_ALIGN] );
    horiz_sub( src_data,
               dst_data,
               shift,
               d_gauss.dd.span[octave],
               &d_gauss.dd.filter[octave*GAUSS_ALIGN] );
}

} // namespace relativeTexAddress

} // namespace variableSpan


__global__
void get_by_2_interpolate( cudaTextureObject_t src_data,
                           const int           src_level,
                           cudaSurfaceObject_t dst_data,
                           const int           dst_w,
                           const int           dst_h )
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if( idx >= dst_w ) return;
    if( idy >= dst_h ) return;

    const float val = tex2DLayered<float>( src_data, 2.0f * idx + 1.0f, 2.0f * idy + 1.0f, src_level );

    surf2DLayeredwrite( val, dst_data, idx*4, idy, 0, cudaBoundaryModeZero ); // dst_data.ptr(idy)[idx] = val;
}

__global__
void get_by_2_pick_every_second( cudaTextureObject_t src_data,
                                 const int           src_w,
                                 const int           src_h,
                                 const int           src_level,
                                 cudaSurfaceObject_t dst_data,
                                 const int           dst_w,
                                 const int           dst_h )
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if( idx >= dst_w ) return;
    if( idy >= dst_h ) return;

    const int read_x = clamp( idx << 1, 0, src_w );
    const int read_y = clamp( idy << 1, 0, src_h );

    const float val = tex2DLayered<float>( src_data, read_x, read_y, src_level );

    surf2DLayeredwrite( val, dst_data, idx*4, idy, 0, cudaBoundaryModeZero ); // dst_data.ptr(idy)[idx] = val;
}


__global__
void make_dog( cudaTextureObject_t src_data,
               cudaSurfaceObject_t dog_data,
               const int           w,
               const int           h,
               const int           level )
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    const int r_x = clamp( idx, w );
    const int r_y = clamp( idy, h );

    const float b = tex2DLayered<float>( src_data, r_x, r_y, level+1 );
    const float a = tex2DLayered<float>( src_data, r_x, r_y, level );
    const float c = b - a;

    surf2DLayeredwrite( c, dog_data, idx*4, idy, level, cudaBoundaryModeZero );
}

} // namespace gauss

__host__
inline void Pyramid::horiz_from_input_image( const Config& conf, Image* base, int octave, cudaStream_t stream, Config::SiftMode mode )
{
    Octave&   oct_obj = _octaves[octave];

    const int width   = oct_obj.getWidth();
    const int height  = oct_obj.getHeight();

    dim3 block( 128, 1 );
    dim3 grid;
    grid.x  = grid_divide( width,  128 );
    grid.y  = height;

    float shift  = 0.5f;

    if( octave == 0 && ( mode == Config::PopSift || mode == Config::VLFeat ) ) {
        shift  = 0.5f * powf( 2.0f, conf.getUpscaleFactor() - octave );
    }

    gauss::variableSpan::relativeTexAddress::horiz
        <<<grid,block,0,stream>>>
        ( base->getInputTexture(),
          oct_obj.getIntermediateData( ),
          octave,
          shift );
}


__host__
inline void Pyramid::downscale_from_prev_octave( int octave, cudaStream_t stream, Config::SiftMode mode )
{
    Octave&      oct_obj = _octaves[octave];
    Octave& prev_oct_obj = _octaves[octave-1];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    /* Necessary to wait for a lower level in the previous octave */
    cudaEvent_t ev = prev_oct_obj.getEventGaussDone( _levels-PREV_LEVEL );
    cudaStreamWaitEvent( stream, ev, 0 );

    dim3 h_block( 64, 2 );
    dim3 h_grid;
    h_grid.x = (unsigned int)grid_divide( width,  h_block.x );
    h_grid.y = (unsigned int)grid_divide( height, h_block.y );

    switch( mode )
    {
    case Config::PopSift :
    case Config::VLFeat :
    case Config::OpenCV :
        gauss::get_by_2_pick_every_second
            <<<h_grid,h_block,0,stream>>>
            ( prev_oct_obj.getDataTexPoint( ),
              prev_oct_obj.getWidth(),
              prev_oct_obj.getHeight(),
              _levels-PREV_LEVEL,
              oct_obj.getDataSurface( ),
              oct_obj.getWidth(),
              oct_obj.getHeight() );
        break;
    default :
        gauss::get_by_2_interpolate
            <<<h_grid,h_block,0,stream>>>
            ( prev_oct_obj.getDataTexLinear( ),
              _levels-PREV_LEVEL,
              oct_obj.getDataSurface( ),
              oct_obj.getWidth(),
              oct_obj.getHeight() );
        break;
    }
}

__host__
inline void Pyramid::horiz_from_prev_level( int octave, int level, cudaStream_t stream )
{
    Octave&      oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    /* waiting for previous level in same octave */
    cudaEvent_t ev = oct_obj.getEventGaussDone( level-1 );
    cudaStreamWaitEvent( stream, ev, 0 );

    dim3 block( 128, 1 );
    dim3 grid;
    grid.x  = grid_divide( width,  128 );
    grid.y  = height;

    gauss::variableSpan::absoluteTexAddress::horiz
        <<<grid,block,0,stream>>>
        ( oct_obj.getDataTexPoint( ),
          oct_obj.getIntermediateData( ),
          oct_obj.getWidth(),
          oct_obj.getHeight(),
          level );
}

__host__
inline void Pyramid::vert_from_interm( int octave, int level, cudaStream_t stream )
{
    Octave& oct_obj = _octaves[octave];

    /* waiting for any events is not necessary, it's in the same stream as horiz
     */

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    dim3 block( 64, 2 );
    dim3 grid;
    grid.x = (unsigned int)grid_divide( width,  block.x );
    grid.y = (unsigned int)grid_divide( height, block.y );

    gauss::variableSpan::absoluteTexAddress::vert
        <<<grid,block,0,stream>>>
        ( oct_obj.getIntermDataTexPoint( ),
          oct_obj.getDataSurface( ),
          oct_obj.getWidth(),
          oct_obj.getHeight(),
          level );
}

__host__
inline void Pyramid::dog_from_blurred( int octave, int level, cudaStream_t stream )
{
    Octave&      oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    dim3 block( 128, 2 );
    dim3 grid;
    grid.x = grid_divide( width,  block.x );
    grid.y = grid_divide( height, block.y );

    /* waiting for lower level is automatic, it's in the same stream.
     * waiting for upper level is necessary, it's in another stream.
     */
    cudaEvent_t  ev     = oct_obj.getEventGaussDone( level-1 );
    cudaStreamWaitEvent( stream, ev, 0 );

    gauss::make_dog
        <<<grid,block,0,stream>>>
        ( oct_obj.getDataTexPoint( ),
          oct_obj.getDogSurface( ),
          oct_obj.getWidth(),
          oct_obj.getHeight(),
          level-1 );
}

/*************************************************************
 * V11: host side
 *************************************************************/
__host__
void Pyramid::build_pyramid( const Config& conf, Image* base )
{
    cudaError_t err;

#if (PYRAMID_PRINT_DEBUG==1)
    cerr << "Entering " << __FUNCTION__ << " with base image "  << endl
         << "    type size         : " << base->type_size << endl
         << "    aligned byte size : " << base->a_width << "x" << base->a_height << endl
         << "    pitch size        : " << base->pitch << "x" << base->a_height << endl
         << "    original byte size: " << base->u_width << "x" << base->u_height << endl
         << "    aligned pix size  : " << base->a_width/base->type_size << "x" << base->a_height << endl
         << "    original pix size : " << base->u_width/base->type_size << "x" << base->u_height << endl;
#endif // (PYRAMID_PRINT_DEBUG==1)

    cudaDeviceSynchronize();

    for( uint32_t octave=0; octave<_num_octaves; octave++ ) {
      Octave& oct_obj   = _octaves[octave];

      if( conf.getGaussMode() == Config::Fixed9 || conf.getGaussMode() == Config::Fixed15 ) {
        cudaStream_t stream = oct_obj.getStream(0);
        if( octave == 0 ) {
            make_octave( conf, base, oct_obj, stream, true );
        } else {
            if( conf.getScalingMode() == Config::ScaleDirect ) {
                horiz_from_input_image( conf, base, octave, stream,
                                        conf.getSiftMode() );
                vert_from_interm( octave, 0, stream );
            } else { // Config::ScaleDefault
                downscale_from_prev_octave( octave, stream,
                                            conf.getSiftMode() );
            }
            make_octave( conf, base, oct_obj, stream, false );
        }

        for( uint32_t level=0; level<_levels; level++ ) {
            cudaEvent_t  ev = oct_obj.getEventGaussDone(level);
            err = cudaEventRecord( ev, stream );
            POP_CUDA_FATAL_TEST( err, "Could not record a Gauss done event: " );
        }

        for( uint32_t level=1; level<_levels; level++ ) {
            cudaEvent_t  dog_ev = oct_obj.getEventDogDone(level);
            err = cudaEventRecord( dog_ev, stream );
            POP_CUDA_FATAL_TEST( err, "Could not record a Gauss done event: " );
        }
      } else {

        for( uint32_t level=0; level<_levels; level++ ) {
            const int width  = oct_obj.getWidth();
            const int height = oct_obj.getHeight();

            cudaStream_t stream = oct_obj.getStream(level);
            cudaEvent_t  ev     = oct_obj.getEventGaussDone(level);
            cudaEvent_t  dog_ev = oct_obj.getEventDogDone(level);

            if( level == 0 )
            {
                if( octave == 0 )
                {
                    horiz_from_input_image( conf, base, 0, stream, conf.getSiftMode() );
                    vert_from_interm( octave, 0, stream );
                }
                else
                {
                    if( conf.getScalingMode() == Config::ScaleDirect ) {
                        horiz_from_input_image( conf, base, octave, stream,
                                                conf.getSiftMode() );
                        vert_from_interm( octave, level, stream );
                    } else { // Config::ScaleDefault
                        downscale_from_prev_octave( octave, stream,
                                                    conf.getSiftMode() );
                    }
                }
            }
            else
            {
                horiz_from_prev_level( octave, level, stream );
                vert_from_interm( octave, level, stream );
            }


            err = cudaEventRecord( ev, stream );
            POP_CUDA_FATAL_TEST( err, "Could not record a Gauss done event: " );

            if( level > 0 ) {
                dog_from_blurred( octave, level, stream );

                err = cudaEventRecord( dog_ev, stream );
                POP_CUDA_FATAL_TEST( err, "Could not record a Gauss done event: " );
            }
        }
      }
    }
}

} // namespace popsift

