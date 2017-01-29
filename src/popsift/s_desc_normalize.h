/*
 * Copyright 2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "sift_extremum.h"

__global__
void normalize_histogram_root_sift( popsift::Descriptor* descs, int num_orientations );

__global__
void normalize_histogram_l2( popsift::Descriptor* descs, int num_orientations );

