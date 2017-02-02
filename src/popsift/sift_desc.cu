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
#include "s_desc_grid.h"
#include "s_desc_igrid.h"
#include "s_desc_pl_grid.h"
#include "s_desc_pl_igrid.h"
#include "assist.h"
#include "common/debug_macros.h"

/*************************************************************
 * V1: device side
 *************************************************************/

using namespace popsift;
using namespace std;

__global__ void ext_desc_loop_starter( int*                featvec_counter,
                                       Extremum*           extrema,
                                       Descriptor*         descs,
                                       int*                feat_to_ext_map,
                                       Plane2D_float       layer,
                                       cudaTextureObject_t layer_tex,
                                       bool                use_root_sift )
{
#if __CUDA_ARCH__ > 350
    if( *featvec_counter == 0 ) return;

    start_ext_desc_loop( featvec_counter,
                         extrema,
                         descs,
                         feat_to_ext_map,
                         layer,
                         layer_tex );

    dim3 grid;
    dim3 block;
    grid.x  = grid_divide( *featvec_counter, 32 );
    block.x = 32;
    block.y = 32;
    block.z = 1;

    if( use_root_sift ) {
        normalize_histogram_root_sift
            <<<grid,block>>>
            ( descs, *featvec_counter );
    } else {
        normalize_histogram_l2
            <<<grid,block>>>
            ( descs, *featvec_counter );
    }
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

    dim3 block;
    dim3 grid;
    grid.x  = grid_divide( *featvec_counter, 32 );
    block.x = 32;
    block.y = 32;
    block.z = 1;

    if( use_root_sift ) {
        normalize_histogram_root_sift
            <<<grid,block>>>
            ( descs, *featvec_counter );
    } else {
        normalize_histogram_l2
            <<<grid,block>>>
            ( descs, *featvec_counter );
    }
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

    dim3 block;
    dim3 grid;
    grid.x  = grid_divide( *featvec_counter, 32 );
    block.x = 32;
    block.y = 32;
    block.z = 1;

    if( use_root_sift ) {
        normalize_histogram_root_sift
            <<<grid,block>>>
            ( descs, *featvec_counter );
    } else {
        normalize_histogram_l2
            <<<grid,block>>>
            ( descs, *featvec_counter );
    }
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

    dim3 block;
    dim3 grid;
    grid.x  = grid_divide( *featvec_counter, 32 );
    block.x = 32;
    block.y = 32;
    block.z = 1;

    if( use_root_sift ) {
        normalize_histogram_root_sift
            <<<grid,block>>>
            ( descs, *featvec_counter );
    } else {
        normalize_histogram_l2
            <<<grid,block>>>
            ( descs, *featvec_counter );
    }
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

    dim3 block;
    dim3 grid;
    grid.x  = grid_divide( *featvec_counter, 32 );
    block.x = 32;
    block.y = 32;
    block.z = 1;

    if( use_root_sift ) {
        normalize_histogram_root_sift
            <<<grid,block>>>
            ( descs, *featvec_counter );
    } else {
        normalize_histogram_l2
            <<<grid,block>>>
            ( descs, *featvec_counter );
    }
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

            for( int level=1; level<_levels-2; level++ ) {
                cudaStream_t oct_str = oct_obj.getStream(level+2);

                if( conf.getDescMode() == Config::Loop ) {
                    ext_desc_loop_starter
                        <<<1,1,0,oct_str>>>
                        ( oct_obj.getFeatVecCtPtrD( level ),
                          oct_obj.getExtrema( level ),
                          oct_obj.getDescriptors( level ),
                          oct_obj.getFeatToExtMapD( level ),
                          oct_obj.getData( level ),
                          oct_obj.getDataTexPoint( level ),
                          conf.getUseRootSift() );
                } else if( conf.getDescMode() == Config::Grid ) {
                    ext_desc_grid_starter
                        <<<1,1,0,oct_str>>>
                        ( oct_obj.getFeatVecCtPtrD( level ),
                          oct_obj.getExtrema( level ),
                          oct_obj.getDescriptors( level ),
                          oct_obj.getFeatToExtMapD( level ),
                          oct_obj.getDataTexPoint( level ),
                          conf.getUseRootSift() );
                } else if( conf.getDescMode() == Config::IGrid ) {
                    ext_desc_igrid_starter
                        <<<1,1,0,oct_str>>>
                        ( oct_obj.getFeatVecCtPtrD( level ),
                          oct_obj.getExtrema( level ),
                          oct_obj.getDescriptors( level ),
                          oct_obj.getFeatToExtMapD( level ),
                          oct_obj.getDataTexLinear( level ),
                          conf.getUseRootSift() );
                } else if( conf.getDescMode() == Config::PLGrid ) {
                    ext_desc_plgrid_starter
                        <<<1,1,0,oct_str>>>
                        ( oct_obj.getFeatVecCtPtrD( level ),
                          oct_obj.getExtrema( level ),
                          oct_obj.getDescriptors( level ),
                          oct_obj.getFeatToExtMapD( level ),
                          oct_obj.getDataTexPoint( level ),
                          conf.getUseRootSift() );
                } else if( conf.getDescMode() == Config::PLIGrid ) {
                    ext_desc_pligrid_starter
                        <<<1,1,0,oct_str>>>
                        ( oct_obj.getFeatVecCtPtrD( level ),
                          oct_obj.getExtrema( level ),
                          oct_obj.getDescriptors( level ),
                          oct_obj.getFeatToExtMapD( level ),
                          oct_obj.getDataTexLinear( level ),
                          conf.getUseRootSift() );
                }
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

            for( int level=3; level<_levels; level++ ) {
                cudaStreamSynchronize( oct_obj.getStream(level) );
            }

            // async copy of extrema from device to host
            oct_obj.readExtremaCount( );
        }

        for( int octave=0; octave<_num_octaves; octave++ ) {
            Octave&      oct_obj = _octaves[octave];


            for( int level=1; level<_levels-2; level++ ) {
                dim3 block;
                dim3 grid;
                grid.x = oct_obj.getFeatVecCountH( level );
                grid.y = 1;
                grid.z = 1;

                if( grid.x != 0 ) {
                    if( conf.getDescMode() == Config::Loop ) {
                        start_ext_desc_loop( oct_obj, level );
                    } else if( conf.getDescMode() == Config::Grid ) {
                        start_ext_desc_grid( oct_obj, level );
                    } else if( conf.getDescMode() == Config::IGrid ) {
                        start_ext_desc_igrid( oct_obj, level );
                    } else if( conf.getDescMode() == Config::PLGrid ) {
                        start_ext_desc_pl_grid( oct_obj, level );
                    } else if( conf.getDescMode() == Config::PLIGrid ) {
                        start_ext_desc_pl_igrid( oct_obj, level );
                    } else {
                        POP_FATAL( "not yet" );
                    }

                    grid.x  = grid_divide( oct_obj.getFeatVecCountH( level ), 32 );
                    block.x = 32;
                    block.y = 32;
                    block.z = 1;

                    if( conf.getUseRootSift() ) {
                        normalize_histogram_root_sift
                            <<<grid,block,0,oct_obj.getStream(level+2)>>>
                            ( oct_obj.getDescriptors( level ),
                              oct_obj.getFeatVecCountH( level ) );
                    } else {
                        normalize_histogram_l2
                            <<<grid,block,0,oct_obj.getStream(level+2)>>>
                            ( oct_obj.getDescriptors( level ),
                              oct_obj.getFeatVecCountH( level ) );
                    }
                }
            }
        }
    }

    cudaDeviceSynchronize( );
}

