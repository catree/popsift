/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <stdio.h>
#include <inttypes.h>

#include "common/plane_2d.h"
#include "sift_constants.h"

/*
 * We are wasting time by computing gradiants on demand several
 * times. We could precompute gradiants for all pixels once, as
 * other code does, but the number of features should be too low
 * to make that feasible. So, we take this performance hit.
 * Especially punishing in the descriptor computation.
 *
 * Also, we are always computing from the closest blur level
 * as Lowe expects us to do. Other implementations compute the
 * gradiant always from the original image, which we think is
 * not in the spirit of the hierarchy is blur levels. That
 * assumption would only hold if we could simply downscale to
 * every first level of every octave ... which is not compatible
 * behaviour.
 */
__device__ static inline
void get_gradiant( float& grad,
                   float& theta,
                   int    x,
                   int    y,
                   popsift::Plane2D_float& layer )
{
    grad  = 0.0f;
    theta = 0.0f;
    if( x > 0 && x < layer.getCols()-1 && y > 0 && y < layer.getRows()-1 ) {
        float dx = layer.ptr(y)[x+1] - layer.ptr(y)[x-1];
        float dy = layer.ptr(y+1)[x] - layer.ptr(y-1)[x];
        grad     = hypotf( dx, dy ); // __fsqrt_rz(dx*dx + dy*dy);
        theta    = atan2f(dy, dx);
    }
}

__device__ static inline
void get_gradiant( float&              grad,
                   float&              theta,
                   const int           x,
                   const int           y,
                   cudaTextureObject_t layer,
                   const int           level )
{
    grad  = 0.0f;
    theta = 0.0f;
    float dx = tex2DLayered<float>( layer, x+1.0f+0.5f, y+0.5f, level ) - tex2DLayered<float>( layer, x-1.0f+0.5f, y+0.5f, level );
    float dy = tex2DLayered<float>( layer, x+0.5f, y+1.0f+0.5f, level ) - tex2DLayered<float>( layer, x+0.5f, y-1.0f+0.5f, level );
    grad     = hypotf( dx, dy ); // __fsqrt_rz(dx*dx + dy*dy);
    theta    = atan2f(dy, dx);
}

// float2 x=grad, y=theta
__device__ static inline
float2 get_gradiant( int x,
                     int y,
                     popsift::Plane2D_float& layer )
{
    if( x > 0 && x < layer.getCols()-1 && y > 0 && y < layer.getRows()-1 ) {
        float dx = layer.ptr(y)[x+1] - layer.ptr(y)[x-1];
        float dy = layer.ptr(y+1)[x] - layer.ptr(y-1)[x];
        return make_float2( hypotf( dx, dy ), // __fsqrt_rz(dx*dx + dy*dy);
                            atan2f(dy, dx) );
    }
    return make_float2( 0.0f, 0.0f );
}

/* The texture-based get_gradiant functions make only sense with a texture in
 * filter mode cudaFilterModePoint
 */
#if 0
__device__ static inline
float gradiant_fetch( cudaTextureObject_t layer, int x, int y )
{
    return tex2D<float>( layer, x+0.5f, y+0.5f );
}

__device__ static inline
void get_gradiant( float& grad,
                   float& theta,
                   int    x,
                   int    y,
                   cudaTextureObject_t layer )
{
    float dx = gradiant_fetch( layer, x+1, y ) - gradiant_fetch( layer, x-1, y );
    float dy = gradiant_fetch( layer, x, y+1 ) - gradiant_fetch( layer, x, y-1 );
    grad     = hypotf( dx, dy );
    theta    = atan2f(dy, dx);
}

__device__ static inline
float2 get_gradiant( int x,
                     int y,
                     cudaTextureObject_t layer )
{
    float dx = gradiant_fetch( layer, x+1, y ) - gradiant_fetch( layer, x-1, y );
    float dy = gradiant_fetch( layer, x, y+1 ) - gradiant_fetch( layer, x, y-1 );
    return make_float2( hypotf( dx, dy ),
                        atan2f(dy, dx) );
}
#endif

/* The float_get_gradiant functions make only sense with a texture in
 * filter mode cudaFilterModeLinear
 */
__device__ static inline
float float_gradiant_fetch( cudaTextureObject_t texLinear, float x, float y, int level )
{
    return tex2DLayered<float>( texLinear, x, y, level );
}

__device__ static inline
void float_get_gradiant( float& grad,
                         float& theta,
                         float    x,
                         float    y,
                         cudaTextureObject_t texLinear,
                         int                 level )
{
    float dx = float_gradiant_fetch( texLinear, x+1.0f, y, level ) - float_gradiant_fetch( texLinear, x-1.0f, y, level );
    float dy = float_gradiant_fetch( texLinear, x, y+1.0f, level ) - float_gradiant_fetch( texLinear, x, y-1.0f, level );
    grad     = hypotf( dx, dy );
    theta    = atan2f(dy, dx);
}

__device__ static inline
float2 float_get_gradiant( float x,
                           float y,
                           cudaTextureObject_t texLinear,
                           int                 level )
{
    float dx = float_gradiant_fetch( texLinear, x+1.0f, y, level ) - float_gradiant_fetch( texLinear, x-1.0f, y, level );
    float dy = float_gradiant_fetch( texLinear, x, y+1.0f, level ) - float_gradiant_fetch( texLinear, x, y-1.0f, level );
    return make_float2( hypotf( dx, dy ),
                        atan2f(dy, dx) );
}

