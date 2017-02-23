/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <iostream>
#include <stdio.h>
#include <iso646.h>

#include "sift_pyramid.h"
#include "sift_constants.h"
#include "s_gradiant.h"
#include "s_desc_normalize.h"
#include "s_desc_loop.h"
#include "s_desc_iloop.h"
#include "s_desc_grid.h"
#include "s_desc_igrid.h"
#include "s_desc_notile.h"
#include "s_desc_pl_grid.h"
#include "s_desc_pl_igrid.h"
#include "common/assist.h"
#include "common/debug_macros.h"

/*************************************************************
 * V1: device side
 *************************************************************/

using namespace popsift;
using namespace std;

#if __CUDA_ARCH__ > 350
__device__ static inline
void ext_norm_starter( int*                featvec_counter,
                       Descriptor*         descs,
                       bool                use_root_sift )
{
    dim3 grid;
    dim3 block;
    grid.x  = grid_divide( *featvec_counter, 32 );
    block.x = 32;
    block.y = 32;
    block.z = 1;

    if( use_root_sift ) {
        normalize_histogram<NormalizeRootSift>
            <<<grid,block>>>
            ( descs, *featvec_counter );
    } else {
        normalize_histogram<NormalizeL2>
            <<<grid,block>>>
            ( descs, *featvec_counter );
    }
}
#endif

__global__ void ext_desc_loop_starter( int*                featvec_counter,
                                       Extremum*           extrema,
                                       Descriptor*         descs,
                                       int*                feat_to_ext_map,
                                       cudaTextureObject_t layer_tex,
                                       const int           w,
                                       const int           h,
                                       bool                use_root_sift )
{
#if __CUDA_ARCH__ > 350
    if( *featvec_counter == 0 ) return;

    start_ext_desc_loop( featvec_counter,
                         extrema,
                         descs,
                         feat_to_ext_map,
                         layer_tex,
                         w,
                         h );

    ext_norm_starter( featvec_counter, descs, use_root_sift );
#else // __CUDA_ARCH__ > 350
    printf( "Dynamic Parallelism requires a card with Compute Capability 3.5 or higher\n" );
#endif // __CUDA_ARCH__ > 350
}

__global__ void ext_desc_iloop_starter( int*                featvec_counter,
                                        Extremum*           extrema,
                                        Descriptor*         descs,
                                        int*                feat_to_ext_map,
                                        cudaTextureObject_t layer_tex,
                                        const int           w,
                                        const int           h,
                                        bool                use_root_sift )
{
#if __CUDA_ARCH__ > 350
    if( *featvec_counter == 0 ) return;

    start_ext_desc_iloop( featvec_counter,
                         extrema,
                         descs,
                         feat_to_ext_map,
                         layer_tex,
                         w,
                         h );

    ext_norm_starter( featvec_counter, descs, use_root_sift );
#else // __CUDA_ARCH__ > 350
    printf( "Dynamic Parallelism requires a card with Compute Capability 3.5 or higher\n" );
#endif // __CUDA_ARCH__ > 350
}

__global__ void ext_desc_grid_starter( int*                featvec_counter,
                                       Extremum*           extrema,
                                       Descriptor*         descs,
                                       int*                feat_to_ext_map,
                                       cudaTextureObject_t layer_tex,
                                       bool                use_root_sift )
{
#if __CUDA_ARCH__ > 350
    if( *featvec_counter == 0 ) return;

    start_ext_desc_grid( featvec_counter,
                         extrema,
                         descs,
                         feat_to_ext_map,
                         layer_tex );

    ext_norm_starter( featvec_counter, descs, use_root_sift );
#else // __CUDA_ARCH__ > 350
    printf( "Dynamic Parallelism requires a card with Compute Capability 3.5 or higher\n" );
#endif // __CUDA_ARCH__ > 350
}

__global__ void ext_desc_igrid_starter( int*                featvec_counter,
                                        Extremum*           extrema,
                                        Descriptor*         descs,
                                        int*                feat_to_ext_map,
                                        cudaTextureObject_t layer_tex,
                                        bool                use_root_sift )
{
#if __CUDA_ARCH__ > 350
    if( *featvec_counter == 0 ) return;

    start_ext_desc_grid( featvec_counter,
                         extrema,
                         descs,
                         feat_to_ext_map,
                         layer_tex );

    ext_norm_starter( featvec_counter, descs, use_root_sift );
#else // __CUDA_ARCH__ > 350
    printf( "Dynamic Parallelism requires a card with Compute Capability 3.5 or higher\n" );
#endif // __CUDA_ARCH__ > 350
}

__global__ void ext_desc_notile_starter( int*                featvec_counter,
                                         Extremum*           extrema,
                                         Descriptor*         descs,
                                         int*                feat_to_ext_map,
                                         cudaTextureObject_t layer_tex,
                                         bool                use_root_sift )
{
#if __CUDA_ARCH__ > 350
    if( *featvec_counter == 0 ) return;

    start_ext_desc_grid( featvec_counter,
                         extrema,
                         descs,
                         feat_to_ext_map,
                         layer_tex );

    ext_norm_starter( featvec_counter, descs, use_root_sift );
#else // __CUDA_ARCH__ > 350
    printf( "Dynamic Parallelism requires a card with Compute Capability 3.5 or higher\n" );
#endif // __CUDA_ARCH__ > 350
}

__global__ void ext_desc_plgrid_starter( int*                featvec_counter,
                                         Extremum*           extrema,
                                         Descriptor*         descs,
                                         int*                feat_to_ext_map,
                                         cudaTextureObject_t layer_tex,
                                         bool                use_root_sift )
{
#if __CUDA_ARCH__ > 350
    if( *featvec_counter == 0 ) return;

    start_ext_desc_pl_grid( featvec_counter,
                            extrema,
                            descs,
                            feat_to_ext_map,
                            layer_tex );

    ext_norm_starter( featvec_counter, descs, use_root_sift );
#else // __CUDA_ARCH__ > 350
    printf( "Dynamic Parallelism requires a card with Compute Capability 3.5 or higher\n" );
#endif // __CUDA_ARCH__ > 350
}
__global__ void ext_desc_pligrid_starter( int*                featvec_counter,
                                         Extremum*           extrema,
                                         Descriptor*         descs,
                                         int*                feat_to_ext_map,
                                         cudaTextureObject_t layer_tex,
                                         bool                use_root_sift )
{
#if __CUDA_ARCH__ > 350
    if( *featvec_counter == 0 ) return;

    start_ext_desc_pl_igrid( featvec_counter,
                             extrema,
                             descs,
                             feat_to_ext_map,
                             layer_tex );

    ext_norm_starter( featvec_counter, descs, use_root_sift );
#else // __CUDA_ARCH__ > 350
    printf( "Dynamic Parallelism requires a card with Compute Capability 3.5 or higher\n" );
#endif // __CUDA_ARCH__ > 350
}

/*************************************************************
 * descriptor extraction
 * TODO: We use the level of the octave in which the keypoint
 *       was found to extract the descriptor. This is
 *       not 100% as intended by Lowe. The paper says:
 *       "magnitudes and gradient are sampled around the
 *        keypoint location, using the scale of the keypoint
 *        to select the level of Gaussian blur for the image."
 *       This implies that a keypoint that has changed octave
 *       in subpixelic refinement is going to be sampled from
 *       the wrong level of the octave.
 *       Unfortunately, we cannot implement getDataTexPoint()
 *       as a layered 2D texture to fix this issue, because that
 *       would require to store blur levels in cudaArrays, which
 *       are hard to write. Alternatively, we could keep a
 *       device-side octave structure that contains an array of
 *       levels on the device side.
 *************************************************************/
__host__
void Pyramid::descriptors( const Config& conf )
{
    if( conf.useDPDescriptors() ) {
        cerr << "Calling descriptors with dynamic parallelism" << endl;

        for( int octave=0; octave<_num_octaves; octave++ ) {
            Octave&      oct_obj = _octaves[octave];

            cudaStream_t oct_str = oct_obj.getStream();

            if( conf.getDescMode() == Config::Loop ) {
                    ext_desc_loop_starter
                        <<<1,1,0,oct_str>>>
                        ( oct_obj.getFeatVecCtPtrD( ),
                          oct_obj.getExtrema( ),
                          oct_obj.getDescriptors( ),
                          oct_obj.getFeatToExtMapD( ),
                          oct_obj.getDataTexPoint( ),
                          oct_obj.getWidth( ),
                          oct_obj.getHeight( ),
                          conf.getUseRootSift() );
            } else if( conf.getDescMode() == Config::ILoop ) {
                    ext_desc_iloop_starter
                        <<<1,1,0,oct_str>>>
                        ( oct_obj.getFeatVecCtPtrD( ),
                          oct_obj.getExtrema( ),
                          oct_obj.getDescriptors( ),
                          oct_obj.getFeatToExtMapD( ),
                          oct_obj.getDataTexLinear( ),
                          oct_obj.getWidth( ),
                          oct_obj.getHeight( ),
                          conf.getUseRootSift() );
            } else if( conf.getDescMode() == Config::Grid ) {
                    ext_desc_grid_starter
                        <<<1,1,0,oct_str>>>
                        ( oct_obj.getFeatVecCtPtrD( ),
                          oct_obj.getExtrema( ),
                          oct_obj.getDescriptors( ),
                          oct_obj.getFeatToExtMapD( ),
                          oct_obj.getDataTexPoint( ),
                          conf.getUseRootSift() );
            } else if( conf.getDescMode() == Config::IGrid ) {
                    ext_desc_igrid_starter
                        <<<1,1,0,oct_str>>>
                        ( oct_obj.getFeatVecCtPtrD( ),
                          oct_obj.getExtrema( ),
                          oct_obj.getDescriptors( ),
                          oct_obj.getFeatToExtMapD( ),
                          oct_obj.getDataTexLinear( ),
                          conf.getUseRootSift() );
            } else if( conf.getDescMode() == Config::NoTile ) {
                    ext_desc_notile_starter
                        <<<1,1,0,oct_str>>>
                        ( oct_obj.getFeatVecCtPtrD( ),
                          oct_obj.getExtrema( ),
                          oct_obj.getDescriptors( ),
                          oct_obj.getFeatToExtMapD( ),
                          oct_obj.getDataTexLinear( ),
                          conf.getUseRootSift() );
            } else if( conf.getDescMode() == Config::PLGrid ) {
                    ext_desc_plgrid_starter
                        <<<1,1,0,oct_str>>>
                        ( oct_obj.getFeatVecCtPtrD( ),
                          oct_obj.getExtrema( ),
                          oct_obj.getDescriptors( ),
                          oct_obj.getFeatToExtMapD( ),
                          oct_obj.getDataTexPoint( ),
                          conf.getUseRootSift() );
            } else if( conf.getDescMode() == Config::PLIGrid ) {
                    ext_desc_pligrid_starter
                        <<<1,1,0,oct_str>>>
                        ( oct_obj.getFeatVecCtPtrD( ),
                          oct_obj.getExtrema( ),
                          oct_obj.getDescriptors( ),
                          oct_obj.getFeatToExtMapD( ),
                          oct_obj.getDataTexLinear( ),
                          conf.getUseRootSift() );
            }
        }

        cudaDeviceSynchronize();

        for( int octave=0; octave<_num_octaves; octave++ ) {
            Octave& oct_obj = _octaves[octave];
            oct_obj.readExtremaCount( );
        }
    } else {
        cerr << "Calling descriptors -no- dynamic parallelism" << endl;

        for( int octave=0; octave<_num_octaves; octave++ ) {
            Octave&      oct_obj = _octaves[octave];

            cudaStreamSynchronize( oct_obj.getStream() );

            // async copy of extrema from device to host
            oct_obj.readExtremaCount( );
        }

        for( int octave=0; octave<_num_octaves; octave++ ) {
            Octave&      oct_obj = _octaves[octave];

            dim3 block;
            dim3 grid;
            grid.x = oct_obj.getFeatVecCountH( );
            grid.y = 1;
            grid.z = 1;

            if( grid.x != 0 ) {
cudaEvent_t desc_start_ev;
cudaEvent_t desc_stop_ev;
if( octave < 2 ) {
cudaEventCreate( &desc_start_ev );
cudaEventCreate( &desc_stop_ev );
cudaEventRecord( desc_start_ev, oct_obj.getStream() );
}
                if( conf.getDescMode() == Config::Loop ) {
                    start_ext_desc_loop( oct_obj );
                } else if( conf.getDescMode() == Config::ILoop ) {
                    start_ext_desc_iloop( oct_obj );
                } else if( conf.getDescMode() == Config::Grid ) {
                    start_ext_desc_grid( oct_obj );
                } else if( conf.getDescMode() == Config::IGrid ) {
                    start_ext_desc_igrid( oct_obj );
                } else if( conf.getDescMode() == Config::NoTile ) {
                    start_ext_desc_notile( oct_obj );
                } else if( conf.getDescMode() == Config::PLGrid ) {
                    start_ext_desc_pl_grid( oct_obj );
                } else if( conf.getDescMode() == Config::PLIGrid ) {
                    start_ext_desc_pl_igrid( oct_obj );
                } else {
                    POP_FATAL( "not yet" );
                }
if( octave < 2 ) {
float ms;
cudaEventRecord( desc_stop_ev, oct_obj.getStream() );
cudaEventSynchronize( desc_stop_ev );
cudaEventElapsedTime( &ms, desc_start_ev, desc_stop_ev );
cudaEventDestroy( desc_start_ev );
cudaEventDestroy( desc_stop_ev );
cerr << "Time for desc in octave " << octave << ": " << setprecision(6) << ms*1000.0f << "us" << endl;
}

                grid.x  = grid_divide( oct_obj.getFeatVecCountH( ), 32 );
                block.x = 32;
                block.y = 32;
                block.z = 1;

                if( conf.getUseRootSift() ) {
                    normalize_histogram<NormalizeRootSift>
                        <<<grid,block,0,oct_obj.getStream( )>>>
                        ( oct_obj.getDescriptors( ),
                          oct_obj.getFeatVecCountH( ) );
                } else {
                    normalize_histogram<NormalizeL2>
                        <<<grid,block,0,oct_obj.getStream( )>>>
                        ( oct_obj.getDescriptors( ),
                          oct_obj.getFeatVecCountH( ) );
                }
            }
        }
    }

    cudaDeviceSynchronize( );
}

