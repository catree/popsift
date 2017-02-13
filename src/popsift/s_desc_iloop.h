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
void ext_desc_iloop( popsift::Extremum*     extrema,
                     popsift::Descriptor*   descs,
                     int*                   feat_to_ext_map,
                     cudaTextureObject_t    layer_tex,
                     const int              width,
                     const int              height );

namespace popsift
{

inline static bool start_ext_desc_iloop( Octave& oct_obj )
{
    dim3 block;
    dim3 grid;
    grid.x = oct_obj.getFeatVecCountH( );
    grid.y = 1;
    grid.z = 1;

    if( grid.x == 0 ) return false;

    block.x = 32;
    block.y = 1;
    block.z = 16;

    ext_desc_iloop
        <<<grid,block,0,oct_obj.getStream()>>>
        ( oct_obj.getExtrema( ),
          oct_obj.getDescriptors( ),
          oct_obj.getFeatToExtMapD( ),
          oct_obj.getDataTexPoint( ),
          oct_obj.getWidth(),
          oct_obj.getHeight() );

    return true;
}

__device__ inline
void start_ext_desc_iloop( int*                featvec_counter,
                          Extremum*           extrema,
                          Descriptor*         descs,
                          int*                feat_to_ext_map,
                          cudaTextureObject_t layer_tex,
                          const int           width,
                          const int           height )
{
#if __CUDA_ARCH__ > 350
    dim3 block;
    dim3 grid;
    grid.x  = *featvec_counter;

    if( grid.x == 0 ) return;

    block.x = 32;
    block.y = 1;
    block.z = 16;

    ext_desc_iloop
        <<<grid,block>>>
        ( extrema,
          descs,
          feat_to_ext_map,
          layer_tex,
          width,
          height );
#endif
}

}; // namespace popsift

