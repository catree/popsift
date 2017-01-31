/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "sift_conf.h"
#include "common/debug_macros.h"

namespace popsift
{

Config::Config( )
    : _upscale_factor( 1.0f )
    , octaves( -1 )
    , levels( 3 )
    , sigma( 1.6f )
    , _edge_limit( 10.0f )
    , _threshold( 0.04 ) // ( 10.0f / 256.0f )
    , _gauss_mode( Config::VLFeat_Compute )
    , _sift_mode( Config::PopSift )
    , _log_mode( Config::None )
    , _scaling_mode( Config::ScaleDefault )
    , _desc_mode( Config::Loop )
    , verbose( false )
    , _max_extrema( 10000 )
    , _assume_initial_blur( true )
    , _initial_blur( 0.5f )
    , _use_root_sift( false )
    , _normalization_multiplier( 0 )
    , _print_gauss_tables( false )
    , _dp_orientation( true )
    , _dp_descriptors( true )
    , _dp_capable( true )
{
    int            currentDev;
    cudaDeviceProp currentProp;
    cudaError_t    err;

    err = cudaGetDevice( &currentDev );
    POP_CUDA_FATAL_TEST( err, "Could not get current device ID" );

    err = cudaGetDeviceProperties( &currentProp, currentDev );
    POP_CUDA_FATAL_TEST( err, "Could not get current device properties" );

    if( currentProp.major < 3 || ( currentProp.major == 3 && currentProp.minor < 5 ) ) {
        _dp_capable = false;
    }
}

void Config::setMode( Config::SiftMode m )
{
    _sift_mode = m;
}

void Config::setGaussMode( Config::GaussMode m )
{
    _gauss_mode = m;
}

void Config::setDescMode( const std::string& text )
{
    if( text == "loop" )
        setDescMode( Config::Loop );
    else if( text == "grid" )
        setDescMode( Config::Grid );
    else if( text == "igrid" )
        setDescMode( Config::IGrid );
    else
        POP_FATAL( "specified descriptor extraction mode must be one of loop, grid or igrid" );
}

void Config::setDescMode( Config::DescMode m )
{
    _desc_mode = m;
}

void Config::setGaussMode( const std::string& m )
{
    if( m == "vlfeat" )
        setGaussMode( Config::VLFeat_Compute );
    else if( m == "opencv" )
        setGaussMode( Config::OpenCV_Compute );
    else if( m == "fixed9" )
        setGaussMode( Config::Fixed9 );
    else if( m == "fixed15" )
        setGaussMode( Config::Fixed15 );
    else
        POP_FATAL( "specified Gauss mode must be one of vlfeat, opencv, fixed9 or fixed15" );
}

void Config::setVerbose( bool on )
{
    verbose = on;
}

void Config::setLogMode( LogMode mode )
{
    _log_mode = mode;
}

Config::LogMode Config::getLogMode( ) const
{
    return _log_mode;
}

void Config::setScalingMode( ScalingMode mode )
{
    _scaling_mode = mode;
}

void Config::setDownsampling( float v ) { _upscale_factor = -v; }
void Config::setOctaves( int v ) { octaves = v; }
void Config::setLevels( int v ) { levels = v; }
void Config::setSigma( float v ) { sigma = v; }
void Config::setEdgeLimit( float v ) { _edge_limit = v; }
void Config::setThreshold( float v ) { _threshold = v; }
void Config::setPrintGaussTables() { _print_gauss_tables = true; }
void Config::setDPOrientation( bool onoff ) { _dp_orientation = onoff; }
void Config::setDPDescriptors( bool onoff ) { _dp_descriptors = onoff; }
void Config::setUseRootSift( bool on ) { _use_root_sift = on; }
void Config::setNormalizationMultiplier( int mul ) { _normalization_multiplier = mul; }

void Config::setInitialBlur( float blur )
{
    if( blur == 0.0f ) {
        _assume_initial_blur = false;
        _initial_blur        = blur;
    } else {
        _assume_initial_blur = true;
        _initial_blur        = blur;
    }
}

Config::GaussMode Config::getGaussMode( ) const
{
    return _gauss_mode;
}

Config::SiftMode Config::getSiftMode() const
{
    return _sift_mode;
}

bool Config::hasInitialBlur( ) const
{
    return _assume_initial_blur;
}

float Config::getInitialBlur( ) const
{
    return _initial_blur;
}

float Config::getPeakThreshold() const
{
    return ( _threshold * 0.5f * 255.0f / levels );
}

bool Config::ifPrintGaussTables() const
{
    return _print_gauss_tables;
}

bool Config::equal( const Config& other ) const
{
    #define COMPARE(a) ( this->a != other.a )
    if( COMPARE( octaves ) ||
        COMPARE( levels ) ||
        COMPARE( sigma ) ||
        COMPARE( _edge_limit ) ||
        COMPARE( _threshold ) ||
        COMPARE( _upscale_factor ) ||
        COMPARE( _scaling_mode ) ||
        COMPARE( _max_extrema ) ||
        COMPARE( _gauss_mode ) ||
        COMPARE( _sift_mode ) ||
        COMPARE( _assume_initial_blur ) ||
        COMPARE( _initial_blur ) ||
        COMPARE( _use_root_sift ) ||
        COMPARE( _normalization_multiplier ) ||
        COMPARE( _dp_orientation ) ||
        COMPARE( _dp_descriptors ) ||
        COMPARE( _dp_capable ) ) return false;
    return true;
}

}; // namespace popsift

