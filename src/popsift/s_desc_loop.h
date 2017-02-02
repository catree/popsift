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
#include "common/plane_2d.h"

__global__
void ext_desc_loop( popsift::Extremum*     extrema,
                    popsift::Descriptor*   descs,
                    int*                   feat_to_ext_map,
                    popsift::Plane2D_float layer,
                    cudaTextureObject_t    layer_tex );

namespace popsift
{

inline static bool start_ext_desc_loop( Octave& oct_obj, int level )
{
    dim3 block;
    dim3 grid;
    grid.x = oct_obj.getFeatVecCountH( level );
    grid.y = 1;
    grid.z = 1;

    if( grid.x == 0 ) return false;

    block.x = 32;
    block.y = 4;
    block.z = 4;

    ext_desc_loop
        <<<grid,block,0,oct_obj.getStream(level+2)>>>
        ( oct_obj.getExtrema( level ),
          oct_obj.getDescriptors( level ),
          oct_obj.getFeatToExtMapD( level ),
          oct_obj.getData( level ),
          oct_obj.getDataTexPoint( level ) );

    return true;
}

__device__ inline
void start_ext_desc_loop( int*                featvec_counter,
                          Extremum*           extrema,
                          Descriptor*         descs,
                          int*                feat_to_ext_map,
                          Plane2D_float       layer,
                          cudaTextureObject_t layer_tex )
{
#if __CUDA_ARCH__ > 350
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
#endif
}

}; // namespace popsift

