/*
* Copyright 2016, Simula Research Laboratory
*
* This Source Code Form is subject to the terms of the Mozilla Public
* License, v. 2.0. If a copy of the MPL was not distributed with this
* file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/
#include <sstream>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#define stat _stat
#define mkdir(name, mode) _mkdir(name)
#endif
#include <new> // for placement new

#include "sift_pyramid.h"
#include "sift_constants.h"
#include "common/debug_macros.h"
#include "common/clamp.h"
#include "common/write_plane_2d.h"
#include "sift_octave.h"

/* Define this only for debugging the descriptor by writing
* the dominent orientation in readable form (otherwise
* incompatible with other tools).
*/
#undef PRINT_WITH_ORIENTATION

using namespace std;

namespace popsift {

    Octave::Octave()
        : _h_extrema_counter(0)
        , _d_extrema_counter(0)
        , _h_featvec_counter(0)
        , _d_featvec_counter(0)
        , _h_extrema(0)
        , _d_extrema(0)
        , _d_desc(0)
        , _h_desc(0)
    { }


    void Octave::alloc(int width, int height, int levels, int gauss_group)
    {
        _w = width;
        _h = height;
        _levels = levels;

        alloc_data_planes();
        alloc_data_tex();

        alloc_interm_plane();
        alloc_interm_tex();

        alloc_dog_array();
        alloc_dog_tex();

        alloc_extrema_mgmt();
        alloc_extrema();

        alloc_streams();
        alloc_events();

        int sz = h_consts.orientations;
        if (sz == 0) {
            _d_desc = 0;
            _h_desc = 0;
        } else {
            _d_desc = popsift::cuda::malloc_devT<Descriptor>(sz, __FILE__, __LINE__);
            _h_desc = popsift::cuda::malloc_hstT<Descriptor>(sz, __FILE__, __LINE__);
        }
    }

    void Octave::free()
    {
        if( _h_desc ) cudaFreeHost( _h_desc );
        if (_d_desc ) cudaFree( _d_desc );

        free_events();
        free_streams();

        free_extrema();
        free_extrema_mgmt();

        free_dog_tex();
        free_dog_array();

        free_interm_tex();
        free_interm_plane();

        free_data_tex();
        free_data_planes();
    }

    void Octave::reset_extrema_mgmt()
    {
        cudaEvent_t  ev = _extrema_done;

        memset(_h_extrema_counter, 0, 1 * sizeof(int));
        popcuda_memset_async(_d_extrema_counter, 0, 1 * sizeof(int), _stream);
        popcuda_memset_async(_d_extrema_num_blocks, 0, 1 * sizeof(int), _stream);
        popcuda_memset_async(_d_featvec_counter, 0, 1 * sizeof(int), _stream);

        cudaEventRecord(ev, _stream);
    }

    void Octave::readExtremaCount()
    {
        popcuda_memcpy_async( _h_extrema_counter,
                              _d_extrema_counter,
                              1 * sizeof(int),
                              cudaMemcpyDeviceToHost,
                              _stream );
        popcuda_memcpy_async( _h_featvec_counter,
                              _d_featvec_counter,
                              1 * sizeof(int),
                              cudaMemcpyDeviceToHost,
                              _stream );
    }

    int Octave::getExtremaCount() const
    {
        return *_h_extrema_counter;
    }

    int Octave::getDescriptorCount() const
    {
        return *_h_featvec_counter;
    }

    void Octave::downloadDescriptor(const Config& conf)
    {
        int sz = *_h_extrema_counter;
        if( sz == 0 ) return;
        popcuda_memcpy_async( _h_extrema,
                              _d_extrema,
                              sz * sizeof(Extremum),
                              cudaMemcpyDeviceToHost,
                              0 );
        sz = *_h_featvec_counter;
        popcuda_memcpy_async( _h_desc,
                              _d_desc,
                              sz * sizeof(Descriptor),
                              cudaMemcpyDeviceToHost,
                              0 );

        cudaDeviceSynchronize();
    }

    void Octave::writeDescriptor(const Config& conf, ostream& ostr, bool really)
    {
        if( _h_extrema == 0 ) return;

        Extremum*   cand = _h_extrema;
        Descriptor* desc = _h_desc;

        int sz = *_h_extrema_counter;

        for (int s = 0; s<sz; s++) {
            for (int ori = 0; ori<cand[s].num_ori; ori++) {
                const float up_fac = conf.getUpscaleFactor();

                float xpos = cand[s].xpos * pow(2.0, _debug_octave_id - up_fac);
                float ypos = cand[s].ypos * pow(2.0, _debug_octave_id - up_fac);
                float sigma = cand[s].sigma * pow(2.0, _debug_octave_id - up_fac);
                float dom_or = cand[s].orientation[ori];
                dom_or = dom_or / M_PI2 * 360;
                if (dom_or < 0) dom_or += 360;

#ifdef PRINT_WITH_ORIENTATION
                ostr << setprecision(5)
                        << xpos << " "
                        << ypos << " "
                        << sigma << " "
                        << dom_or << " ";
#else
                ostr << setprecision(5)
                        << xpos << " " << ypos << " "
                        << 1.0f / (sigma * sigma)
                        << " 0 "
                        << 1.0f / (sigma * sigma) << " ";
#endif
                if (really) {
                        int feat_vec_index = cand[s].idx_ori + ori;
                        for (int i = 0; i<128; i++) {
                            ostr << desc[feat_vec_index].features[i] << " ";
                        }
                }
                ostr << endl;
            }
        }
    }

    void Octave::copyExtrema(const Config& conf, Feature* feature, Descriptor* descBuffer)
    {
        int num_extrema     = getExtremaCount();
        int num_descriptors = getDescriptorCount();

        Extremum*   ext     = _h_extrema;
        Descriptor* desc    = _h_desc;
        int         ext_sz  = *_h_extrema_counter;
        int         desc_sz = *_h_featvec_counter;

        memcpy(descBuffer, desc, desc_sz * sizeof(Descriptor));
        for (int i = 0; i<ext_sz; i++) {
                const float up_fac = conf.getUpscaleFactor();

                float xpos = ext[i].xpos * pow(2.0, _debug_octave_id - up_fac);
                float ypos = ext[i].ypos * pow(2.0, _debug_octave_id - up_fac);
                float sigma = ext[i].sigma * pow(2.0, _debug_octave_id - up_fac);
                int   num_ori = ext[i].num_ori;

                feature[i].xpos = xpos;
                feature[i].ypos = ypos;
                feature[i].sigma = sigma;
                feature[i].num_descs = num_ori;


                int ori;
                for (ori = 0; ori<num_ori; ori++) {
                    int desc_idx = ext[i].idx_ori + ori;
                    feature[i].orientation[ori] = ext[i].orientation[ori];
                    feature[i].desc[ori] = &descBuffer[desc_idx];
                }
                for (; ori<ORIENTATION_MAX_COUNT; ori++) {
                    feature[i].orientation[ori] = 0;
                    feature[i].desc[ori] = 0;
                }
        }

        feature += ext_sz;
        descBuffer += desc_sz;
    }

    Descriptor* Octave::getDescriptors( )
    {
        return _d_desc;
    }

    /*************************************************************
    * Debug output: write an octave/level to disk as PGM
    *************************************************************/

    void Octave::download_and_save_array( const char* basename, int octave )
    {
        // cerr << "Calling " << __FUNCTION__ << " for octave " << octave << endl;

        struct stat st = { 0 };

#if 0
        {
            if (level == 0) {
                int width  = getWidth();
                int height = getHeight();

                Plane2D_float hostPlane_f;
                hostPlane_f.allocHost(width, height, CudaAllocated);
                hostPlane_f.memcpyFromDevice(getData(level));

                uint32_t total_ct = 0;

                readExtremaCount();
                cudaDeviceSynchronize();
                for (uint32_t l = 0; l<_levels; l++) {
                    uint32_t ct = getExtremaCountH(l); // getExtremaCount( l );
                    if (ct > 0) {
                        total_ct += ct;

                        Extremum* cand = new Extremum[ct];

                        popcuda_memcpy_sync(cand,
                            _d_extrema[l],
                            ct * sizeof(Extremum),
                            cudaMemcpyDeviceToHost);
                        for (uint32_t i = 0; i<ct; i++) {
                            int32_t x = roundf(cand[i].xpos);
                            int32_t y = roundf(cand[i].ypos);
                            // cerr << "(" << x << "," << y << ") scale " << cand[i].sigma << " orient " << cand[i].orientation << endl;
                            for (int32_t j = -4; j <= 4; j++) {
                                hostPlane_f.ptr(clamp(y + j, height))[clamp(x, width)] = 255;
                                hostPlane_f.ptr(clamp(y, height))[clamp(x + j, width)] = 255;
                            }
                        }

                        delete[] cand;
                    }
                }

                if (total_ct > 0) {
                    if (stat("dir-feat", &st) == -1) {
                        mkdir("dir-feat", 0700);
                    }

                    if (stat("dir-feat-txt", &st) == -1) {
                        mkdir("dir-feat-txt", 0700);
                    }


                    ostringstream ostr;
                    ostr << "dir-feat/" << basename << "-o-" << octave << "-l-" << level << ".pgm";
                    ostringstream ostr2;
                    ostr2 << "dir-feat-txt/" << basename << "-o-" << octave << "-l-" << level << ".txt";

                    popsift::write_plane2D(ostr.str().c_str(), false, hostPlane_f);
                    popsift::write_plane2Dunscaled(ostr2.str().c_str(), false, hostPlane_f);
                }

                hostPlane_f.freeHost(CudaAllocated);
            }
        }
#endif

            cudaError_t err;
            int width  = getWidth();
            int height = getHeight();

            if (stat("dir-octave", &st) == -1) {
                mkdir("dir-octave", 0700);
            }

            if (stat("dir-octave-dump", &st) == -1) {
                mkdir("dir-octave-dump", 0700);
            }

            if (stat("dir-dog", &st) == -1) {
                mkdir("dir-dog", 0700);
            }

            if (stat("dir-dog-txt", &st) == -1) {
                mkdir("dir-dog-txt", 0700);
            }

            if (stat("dir-dog-dump", &st) == -1) {
                mkdir("dir-dog-dump", 0700);
            }

            float* array;
            POP_CUDA_MALLOC_HOST(&array, width * height * _levels * sizeof(float));

            cudaMemcpy3DParms s = { 0 };
            memset( &s, 0, sizeof(cudaMemcpy3DParms) );
            s.srcArray = _data;
            s.dstPtr   = make_cudaPitchedPtr( array, width * sizeof(float), width, height );
            s.extent   = make_cudaExtent( width, height, _levels );
            s.kind     = cudaMemcpyDeviceToHost;
            err = cudaMemcpy3D(&s);
            POP_CUDA_FATAL_TEST(err, "cudaMemcpy3D failed: ");

            for( int l = 0; l<_levels; l++ ) {
                Plane2D_float p(width, height, &array[l*width*height], width * sizeof(float));

                ostringstream ostr;
                ostr << "dir-octave/" << basename << "-o-" << octave << "-l-" << l << ".pgm";
                popsift::write_plane2Dunscaled( ostr.str().c_str(), false, p );

                ostringstream ostr2;
                ostr2 << "dir-octave-dump/" << basename << "-o-" << octave << "-l-" << l << ".dump";
                popsift::dump_plane2Dfloat(ostr2.str().c_str(), false, p );
            }

            memset( &s, 0, sizeof(cudaMemcpy3DParms) );
            s.srcArray = _dog_3d;
            s.dstPtr = make_cudaPitchedPtr(array, width * sizeof(float), width, height);
            s.extent = make_cudaExtent(width, height, _levels - 1);
            s.kind = cudaMemcpyDeviceToHost;
            err = cudaMemcpy3D(&s);
            POP_CUDA_FATAL_TEST(err, "cudaMemcpy3D failed: ");

            for (int l = 0; l<_levels - 1; l++) {
                Plane2D_float p(width, height, &array[l*width*height], width * sizeof(float));

                ostringstream ostr;
                ostr << "dir-dog/d-" << basename << "-o-" << octave << "-l-" << l << ".pgm";
                popsift::write_plane2D(ostr.str().c_str(), false, p);

                ostringstream pstr;
                pstr << "dir-dog-txt/d-" << basename << "-o-" << octave << "-l-" << l << ".txt";
                popsift::write_plane2Dunscaled(pstr.str().c_str(), false, p, 127);

                ostringstream qstr;
                qstr << "dir-dog-dump/d-" << basename << "-o-" << octave << "-l-" << l << ".dump";
                popsift::dump_plane2Dfloat(qstr.str().c_str(), false, p);
            }

            POP_CUDA_FREE_HOST(array);
    }

    void Octave::alloc_data_planes()
    {
        cudaError_t err;

        _data_desc.f = cudaChannelFormatKindFloat;
        _data_desc.x = 32;
        _data_desc.y = 0;
        _data_desc.z = 0;
        _data_desc.w = 0;

        _data_ext.width  = _w; // for cudaMalloc3DArray, width in elements
        _data_ext.height = _h;
        _data_ext.depth  = _levels;

        err = cudaMalloc3DArray( &_data,
                                 &_data_desc,
                                 _data_ext,
                                 cudaArrayLayered | cudaArraySurfaceLoadStore);
        POP_CUDA_FATAL_TEST(err, "Could not allocate Blur level array: ");
    }

    void Octave::free_data_planes()
    {
        cudaError_t err;

        err = cudaFreeArray( _data );
        POP_CUDA_FATAL_TEST(err, "Could not free Blur level array: ");
    }

    void Octave::alloc_data_tex()
    {
        cudaError_t err;

        cudaResourceDesc res_desc;
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = _data;

        err = cudaCreateSurfaceObject(&_data_surf, &res_desc);
        POP_CUDA_FATAL_TEST(err, "Could not create Blur data surface: ");

        cudaTextureDesc      tex_desc;

        memset(&tex_desc, 0, sizeof(cudaTextureDesc));
        tex_desc.normalizedCoords = 0; // addressed (x,y) in [width,height]
        tex_desc.addressMode[0]   = cudaAddressModeClamp;
        tex_desc.addressMode[1]   = cudaAddressModeClamp;
        tex_desc.addressMode[2]   = cudaAddressModeClamp;
        tex_desc.readMode         = cudaReadModeElementType; // read as float
        tex_desc.filterMode       = cudaFilterModePoint; // no interpolation

        err = cudaCreateTextureObject( &_data_tex_point, &res_desc, &tex_desc, 0 );
        POP_CUDA_FATAL_TEST(err, "Could not create Blur data point texture: ");

        memset(&tex_desc, 0, sizeof(cudaTextureDesc));
        tex_desc.normalizedCoords = 0; // addressed (x,y) in [width,height]
        tex_desc.addressMode[0]   = cudaAddressModeClamp;
        tex_desc.addressMode[1]   = cudaAddressModeClamp;
        tex_desc.addressMode[2]   = cudaAddressModeClamp;
        tex_desc.readMode         = cudaReadModeElementType; // read as float
        tex_desc.filterMode       = cudaFilterModeLinear; // no interpolation

        err = cudaCreateTextureObject( &_data_tex_linear, &res_desc, &tex_desc, 0 );
        POP_CUDA_FATAL_TEST(err, "Could not create Blur data point texture: ");
    }

    void Octave::free_data_tex()
    {
        cudaError_t err;

        err = cudaDestroyTextureObject(_data_tex_point);
        POP_CUDA_FATAL_TEST(err, "Could not destroy Blur data point texture: ");

        err = cudaDestroyTextureObject(_data_tex_linear);
        POP_CUDA_FATAL_TEST(err, "Could not destroy Blur data linear texture: ");

        err = cudaDestroySurfaceObject(_data_surf);
        POP_CUDA_FATAL_TEST(err, "Could not destroy Blur data surface: ");
    }

    void Octave::alloc_interm_plane()
    {
        _intermediate_data.allocDev(_w, _h);
    }

    void Octave::free_interm_plane()
    {
        _intermediate_data.freeDev();
    }

    void Octave::alloc_interm_tex()
    {
        cudaError_t err;

        cudaTextureDesc      interm_data_tex_desc;
        cudaResourceDesc     interm_data_res_desc;

        memset(&interm_data_tex_desc, 0, sizeof(cudaTextureDesc));
        interm_data_tex_desc.normalizedCoords = 0; // addressed (x,y) in [width,height]
        interm_data_tex_desc.addressMode[0] = cudaAddressModeClamp;
        interm_data_tex_desc.addressMode[1] = cudaAddressModeClamp;
        interm_data_tex_desc.addressMode[2] = cudaAddressModeClamp;
        interm_data_tex_desc.readMode = cudaReadModeElementType; // read as float
        interm_data_tex_desc.filterMode = cudaFilterModePoint;

        memset(&interm_data_res_desc, 0, sizeof(cudaResourceDesc));
        interm_data_res_desc.resType = cudaResourceTypePitch2D;
        interm_data_res_desc.res.pitch2D.desc.f = cudaChannelFormatKindFloat;
        interm_data_res_desc.res.pitch2D.desc.x = 32;
        interm_data_res_desc.res.pitch2D.desc.y = 0;
        interm_data_res_desc.res.pitch2D.desc.z = 0;
        interm_data_res_desc.res.pitch2D.desc.w = 0;

        interm_data_res_desc.res.pitch2D.devPtr = _intermediate_data.data;
        interm_data_res_desc.res.pitch2D.pitchInBytes = _intermediate_data.step;
        interm_data_res_desc.res.pitch2D.width = _intermediate_data.getCols();
        interm_data_res_desc.res.pitch2D.height = _intermediate_data.getRows();

        err = cudaCreateTextureObject(&_interm_data_tex,
            &interm_data_res_desc,
            &interm_data_tex_desc, 0);
        POP_CUDA_FATAL_TEST(err, "Could not create texture object: ");
    }

    void Octave::free_interm_tex()
    {
        cudaError_t err;

        err = cudaDestroyTextureObject(_interm_data_tex);
        POP_CUDA_FATAL_TEST(err, "Could not destroy texture object: ");
    }

    void Octave::alloc_dog_array()
    {
        cudaError_t err;

        _dog_3d_desc.f = cudaChannelFormatKindFloat;
        _dog_3d_desc.x = 32;
        _dog_3d_desc.y = 0;
        _dog_3d_desc.z = 0;
        _dog_3d_desc.w = 0;

        _dog_3d_ext.width = _w; // for cudaMalloc3DArray, width in elements
        _dog_3d_ext.height = _h;
        _dog_3d_ext.depth = _levels - 1;

        err = cudaMalloc3DArray(&_dog_3d,
            &_dog_3d_desc,
            _dog_3d_ext,
            cudaArrayLayered | cudaArraySurfaceLoadStore);
        POP_CUDA_FATAL_TEST(err, "Could not allocate 3D DoG array: ");
    }

    void Octave::free_dog_array()
    {
        cudaError_t err;

        err = cudaFreeArray(_dog_3d);
        POP_CUDA_FATAL_TEST(err, "Could not free 3D DoG array: ");
    }

    void Octave::alloc_dog_tex()
    {
        cudaError_t err;

        cudaResourceDesc dog_res_desc;
        dog_res_desc.resType = cudaResourceTypeArray;
        dog_res_desc.res.array.array = _dog_3d;

        err = cudaCreateSurfaceObject(&_dog_3d_surf, &dog_res_desc);
        POP_CUDA_FATAL_TEST(err, "Could not create DoG surface: ");

        cudaTextureDesc      dog_tex_desc;
        memset(&dog_tex_desc, 0, sizeof(cudaTextureDesc));
        dog_tex_desc.normalizedCoords = 0; // addressed (x,y) in [width,height]
        dog_tex_desc.addressMode[0] = cudaAddressModeClamp;
        dog_tex_desc.addressMode[1] = cudaAddressModeClamp;
        dog_tex_desc.addressMode[2] = cudaAddressModeClamp;
        dog_tex_desc.readMode = cudaReadModeElementType; // read as float
        dog_tex_desc.filterMode = cudaFilterModePoint; // no interpolation

        err = cudaCreateTextureObject(&_dog_3d_tex_point, &dog_res_desc, &dog_tex_desc, 0);
        POP_CUDA_FATAL_TEST(err, "Could not create DoG texture: ");

        dog_tex_desc.filterMode = cudaFilterModeLinear; // linear interpolation
        err = cudaCreateTextureObject(&_dog_3d_tex_linear, &dog_res_desc, &dog_tex_desc, 0);
        POP_CUDA_FATAL_TEST(err, "Could not create DoG texture: ");
    }

    void Octave::free_dog_tex()
    {
        cudaError_t err;

        err = cudaDestroyTextureObject(_dog_3d_tex_linear);
        POP_CUDA_FATAL_TEST(err, "Could not destroy DoG texture: ");

        err = cudaDestroyTextureObject(_dog_3d_tex_point);
        POP_CUDA_FATAL_TEST(err, "Could not destroy DoG texture: ");

        err = cudaDestroySurfaceObject(_dog_3d_surf);
        POP_CUDA_FATAL_TEST(err, "Could not destroy DoG surface: ");
    }

    void Octave::alloc_extrema_mgmt()
    {
        _h_extrema_counter = popsift::cuda::malloc_hstT<int>(1, __FILE__, __LINE__);
        _d_extrema_counter = popsift::cuda::malloc_devT<int>(1, __FILE__, __LINE__);
        _d_extrema_num_blocks = popsift::cuda::malloc_devT<int>(1, __FILE__, __LINE__);
        _h_featvec_counter = popsift::cuda::malloc_hstT<int>(1, __FILE__, __LINE__);
        _d_featvec_counter = popsift::cuda::malloc_devT<int>(1, __FILE__, __LINE__);
    }

    void Octave::free_extrema_mgmt()
    {
        cudaFree(     _d_extrema_num_blocks );
        cudaFree(     _d_extrema_counter );
        cudaFreeHost( _h_extrema_counter );
        cudaFree(     _d_featvec_counter );
        cudaFreeHost( _h_featvec_counter );
    }

    void Octave::alloc_extrema()
    {
        _d_extrema = popsift::cuda::malloc_devT<Extremum>( h_consts.extrema,
                                                           __FILE__, __LINE__);
        _h_extrema = popsift::cuda::malloc_hstT<Extremum>( h_consts.extrema,
                                                           __FILE__, __LINE__);
        _d_feat_to_ext_map = popsift::cuda::malloc_devT<int>( h_consts.orientations,
                                                              __FILE__, __LINE__);
        _h_feat_to_ext_map = popsift::cuda::malloc_hstT<int>( h_consts.orientations,
                                                              __FILE__, __LINE__);
    }

    void Octave::free_extrema()
    {
        cudaFreeHost( _h_feat_to_ext_map );
        cudaFree(     _d_feat_to_ext_map );
        cudaFreeHost( _h_extrema );
        cudaFree(     _d_extrema );
    }

    void Octave::alloc_streams()
    {
        _stream = popsift::cuda::stream_create(__FILE__, __LINE__);
    }

    void Octave::free_streams()
    {
        popsift::cuda::stream_destroy( _stream, __FILE__, __LINE__ );
    }

    void Octave::alloc_events()
    {
        _dog_done     = popsift::cuda::event_create(__FILE__, __LINE__);
        _extrema_done = popsift::cuda::event_create(__FILE__, __LINE__);
    }

    void Octave::free_events()
    {
        popsift::cuda::event_destroy( _dog_done,     __FILE__, __LINE__);
        popsift::cuda::event_destroy( _extrema_done, __FILE__, __LINE__);
    }

} // namespace popsift
