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

#undef IGRID_XDIM_IS_32

/*
 * We assume that this is started with
 * block = 16,4,4 or with 32,4,4, depending on macros
 * grid  = nunmber of orientations
 */
__global__
void ext_desc_igrid( popsift::Extremum*     extrema,
                     popsift::Descriptor*   descs,
                     int*                   feat_to_ext_map,
                     cudaTextureObject_t    texLinear,
                     int                    level );

namespace popsift
{

inline static bool start_ext_desc_igrid( Octave& oct_obj, int level )
{
    dim3 block;
    dim3 grid;
    grid.x = oct_obj.getFeatVecCountH( level );
    grid.y = 1;
    grid.z = 1;

    if( grid.x == 0 ) return false;

#ifdef IGRID_XDIM_IS_32
    block.x = 16;
    block.y = 4;
    block.z = 4;
#else
    block.x = 32;
    block.y = 4;
    block.z = 4;
#endif

    ext_desc_igrid
        <<<grid,block,0,oct_obj.getStream(level+2)>>>
        ( oct_obj.getExtrema( level ),
          oct_obj.getDescriptors( level ),
          oct_obj.getFeatToExtMapD( level ),
          oct_obj.getDataTexLinear( ),
          level );

    return true;
}

__device__ inline
void start_ext_desc_igrid( int*                featvec_counter,
                           Extremum*           extrema,
                           Descriptor*         descs,
                           int*                feat_to_ext_map,
                           cudaTextureObject_t texLinear,
                           int                 level )
{
#if __CUDA_ARCH__ > 350
    dim3 block;
    dim3 grid;
    grid.x  = *featvec_counter;

    if( grid.x == 0 ) return;

#ifdef IGRID_XDIM_IS_32
    block.x = 16;
    block.y = 4;
    block.z = 4;
#else
    block.x = 32;
    block.y = 4;
    block.z = 4;
#endif

    ext_desc_igrid
        <<<grid,block>>>
        ( extrema,
          descs,
          feat_to_ext_map,
          texLinear,
          level );
#endif
}

}; // namespace popsift
