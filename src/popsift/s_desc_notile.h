/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once
#include "sift_octave.h"
#include "sift_extremum.h"

/*
 * We assume that this is started with
 * block = 16,4,4 or with 32,4,4, depending on macros
 * grid  = nunmber of orientations
 */
__global__
void ext_desc_notile( popsift::Extremum*     extrema,
                      const int              num,
                      popsift::Descriptor*   descs,
                      int*                   feat_to_ext_map,
                      cudaTextureObject_t    texLinear );

namespace popsift
{

#define NUMLINES 2

inline static bool start_ext_desc_notile( Octave& oct_obj )
{
    dim3 block;
    dim3 grid;
    grid.x = grid_divide( oct_obj.getFeatVecCountH( ), NUMLINES );
    grid.y = 1;
    grid.z = 1;

    if( grid.x == 0 ) return false;

    block.x = 32;
    block.y = 1;
    block.z = NUMLINES;

    ext_desc_notile
        <<<grid,block,0,oct_obj.getStream()>>>
        ( oct_obj.getExtrema( ),
          oct_obj.getFeatVecCountH( ),
          oct_obj.getDescriptors( ),
          oct_obj.getFeatToExtMapD( ),
          oct_obj.getDataTexLinear( ) );

    return true;
}

__device__ inline
void start_ext_desc_notile( int*                featvec_counter,
                            Extremum*           extrema,
                            Descriptor*         descs,
                            int*                feat_to_ext_map,
                            cudaTextureObject_t texLinear )
{
#if __CUDA_ARCH__ > 350
    dim3 block;
    dim3 grid;
    grid.x  = grid_divide( *featvec_counter, NUMLINES );

    if( grid.x == 0 ) return;

    block.x = 32;
    block.y = 1;
    block.z = NUMLINES;

    ext_desc_notile
        <<<grid,block>>>
        ( extrema,
          *featvec_counter,
          descs,
          feat_to_ext_map,
          texLinear );
#endif
}

}; // namespace popsift
