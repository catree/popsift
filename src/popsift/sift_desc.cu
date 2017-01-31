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
#include "assist.h"
#include "common/debug_macros.h"

/*************************************************************
 * V1: device side
 *************************************************************/

using namespace popsift;
using namespace std;

#if __CUDA_ARCH__ > 350
__global__ void descriptor_starter( int*                extrema_counter,
                                    int*                featvec_counter,
                                    Extremum*           extrema,
                                    Descriptor*         descs,
                                    int*                feat_to_ext_map,
                                    Plane2D_float       layer,
                                    cudaTextureObject_t layer_tex,
                                    bool                use_root_sift )
{
    dim3 block;
    dim3 grid;
    grid.x  = *featvec_counter;

    if( grid.x == 0 ) return;

    block.x = 32;
    block.y = 4;
    block.z = 4;

    ext_desc_loop
        <<<grid,block>>>
        ( extrema,
          descs,
          feat_to_ext_map,
          layer,
          layer_tex );

    // it may be good to start more threads, but this kernel
    // is too fast to be noticable in profiling

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
}
#else // __CUDA_ARCH__ > 350
__global__ void descriptor_starter( int*                extrema_counter,
                                    int*                featvec_counter,
                                    Extremum*           extrema,
                                    Descriptor*         descs,
                                    int*                feat_to_ext_map,
                                    Plane2D_float       layer,
                                    cudaTextureObject_t layer_tex,
                                    bool                use_root_sift )
{
    printf( "Dynamic Parallelism requires a card with Compute Capability 3.5 or higher\n" );
}
#endif // __CUDA_ARCH__ > 350

/*************************************************************
 * V4: host side
 *************************************************************/
__host__
void Pyramid::descriptors( const Config& conf )
{
    if( conf.useDPDescriptors() ) {
        // cerr << "Calling descriptors with dynamic parallelism" << endl;

        for( int octave=0; octave<_num_octaves; octave++ ) {
            Octave&      oct_obj = _octaves[octave];

            for( int level=1; level<_levels-2; level++ ) {
                cudaStream_t oct_str = oct_obj.getStream(level+2);

                descriptor_starter
                    <<<1,1,0,oct_str>>>
                    ( oct_obj.getExtremaCtPtrD( level ),
                      oct_obj.getFeatVecCtPtrD( level ),
                      oct_obj.getExtrema( level ),
                      oct_obj.getDescriptors( level ),
                      oct_obj.getFeatToExtMapD( level ),
                      oct_obj.getData( level ),
                      oct_obj._data_tex[level],
                      conf.getUseRootSift() );
            }
        }

        cudaDeviceSynchronize();

        for( int octave=0; octave<_num_octaves; octave++ ) {
            Octave& oct_obj = _octaves[octave];
            oct_obj.readExtremaCount( );
        }
    } else {
        // cerr << "Calling descriptors -no- dynamic parallelism" << endl;

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

                if( grid.x != 0 ) {
                    if( conf.getDescMode() == Config::Loop ) {
                        block.x = 32;
                        block.y = 4;
                        block.z = 4;

                        ext_desc_loop
                            <<<grid,block,0,oct_obj.getStream(level+2)>>>
                            ( oct_obj.getExtrema( level ),
                              oct_obj.getDescriptors( level ),
                              oct_obj.getFeatToExtMapD( level ),
                              oct_obj.getData( level ),
                              oct_obj._data_tex[level] );
                    } else if( conf.getDescMode() == Config::Grid ) {
                        block.x = 16;
                        block.y = 4;
                        block.z = 4;

                        ext_desc_grid
                            <<<grid,block,0,oct_obj.getStream(level+2)>>>
                            ( oct_obj.getExtrema( level ),
                              oct_obj.getDescriptors( level ),
                              oct_obj.getFeatToExtMapD( level ),
                              oct_obj.getData( level ),
                              oct_obj._data_tex[level] );
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

