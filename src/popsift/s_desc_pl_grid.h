/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once
#include "sift_extremum.h"
#include "common/plane_2d.h"

/*
 * We assume that this is started with
 * block = 16,4,4
 * grid  = nunmber of orientations
 */
__global__
void ext_desc_pl_grid( popsift::Extremum*     extrema,
                       popsift::Descriptor*   descs,
                       int*                   feat_to_ext_map,
                       cudaTextureObject_t    layer_tex );

