#pragma once

#include <immintrin.h>
#include <complex>

#include "common.hpp"

namespace sum_man {
    
namespace avx {    

    template <class T>
    struct Treg;

    template <>
    struct Treg<std::complex<double>> {
        typedef __m256d val;
    };
    
    template <>
    struct Treg<double> {
        typedef __m256d val;
    };
    
    template <>
    struct Treg<std::complex<float>> {
        typedef __m256 val;
    };
    
    template <>
    struct Treg<float> {
        typedef __m256 val;
    };
    
    
    template <class T>
    struct Ops;
    
    template <>
    struct Ops<double> {
        static force_inline __m256d unpack(double *arr) {
            return _mm256_loadu_pd(arr);
        }
        static force_inline __m256d unpack(double *arr0, double *arr1) {
            return _mm256_setr_pd(arr0[0], arr0[1], arr1[0], arr1[1]);
        }
        static force_inline __m256d unpack(double *arr0, double *arr1, double *arr2, double *arr3) {
            return _mm256_setr_pd(arr0[0], arr1[0], arr2[0], arr3[0]);
        }

        static force_inline void pack( __m256d src, double *arr) {
            _mm256_storeu_pd(arr, src);
        }        
        static force_inline void pack( __m256d src, double *arr0, double *arr1) {
            double* result = (double *)&src;
            arr0[0] = result[0]; arr0[1] = result[1];
            arr1[0] = result[2]; arr1[1] = result[3];
        }        
        static force_inline void pack( __m256d src, double *arr0, double *arr1, double *arr2, double *arr3) {
            double* result = (double *)&src;
            arr0[0] = result[0];
            arr1[0] = result[1];
            arr2[0] = result[2];
            arr3[0] = result[3];
        }        
    };

    template <>
    struct Ops<std::complex<double>> {
        static force_inline __m256d unpack(std::complex<double> *arr) {
            return Ops<double>::unpack((double *)arr);
        }
        static force_inline __m256d unpack(std::complex<double> *arr0, std::complex<double> *arr1) {
            return Ops<double>::unpack((double *)arr0, (double *)arr1);
        }

        static force_inline void pack( __m256d src, std::complex<double> *arr) {
            Ops<double>::pack(src, (double *)arr);
        }        
        static force_inline void pack( __m256d src, std::complex<double> *arr0, std::complex<double> *arr1) {
            Ops<double>::pack(src, (double *)arr0, (double *)arr1);
        }        
    };
    
    template <>
    struct Ops<float> {
        static force_inline __m256 unpack(float *arr) {
            return _mm256_loadu_ps(arr);
        }
        static force_inline __m256 unpack(float *arr0, float *arr1) {
            return _mm256_setr_ps(arr0[0], arr0[1], arr0[2], arr0[3], arr1[0], arr1[1], arr1[2], arr1[3]);
        }
        static force_inline __m256 unpack(float *arr0, float *arr1, float *arr2, float *arr3) {
            return _mm256_setr_ps(arr0[0], arr0[1], arr1[0], arr1[1], arr2[0], arr2[1], arr3[0], arr3[1]);
        }
        static force_inline __m256 unpack(float *arr0, float *arr1, float *arr2, float *arr3, float *arr4, float *arr5, float *arr6, float *arr7) {
            return _mm256_setr_ps(arr0[0], arr1[0], arr2[0], arr3[0], arr4[0], arr5[0], arr6[0], arr7[0]);
        }

        static force_inline void pack( __m256 src, float *arr) {
            _mm256_storeu_ps(arr, src);
        }        
        static force_inline void pack( __m256 src, float *arr0, float *arr1) {
            float* result = (float*)&src;
            arr0[0] = result[0]; arr0[1] = result[1]; arr0[2] = result[2]; arr0[3] = result[3];
            arr1[0] = result[4]; arr1[1] = result[5]; arr1[2] = result[6]; arr1[3] = result[7];
        }        
        static force_inline void pack( __m256 src, float *arr0, float *arr1, float *arr2, float *arr3) {
            float* result = (float*)&src;
            arr0[0] = result[0]; arr0[1] = result[1];
            arr1[0] = result[2]; arr1[1] = result[3];
            arr2[0] = result[4]; arr2[1] = result[5];
            arr3[0] = result[6]; arr3[1] = result[7];
        }        
        static force_inline void pack( __m256 src, float *arr0, float *arr1, float *arr2, float *arr3, float *arr4, float *arr5, float *arr6, float *arr7) {
            float* result = (float*)&src;
            arr0[0] = result[0];
            arr1[0] = result[1];
            arr2[0] = result[2];
            arr3[0] = result[3];
            arr4[0] = result[4];
            arr5[0] = result[5];
            arr6[0] = result[6];
            arr7[0] = result[7];
        }        
    };

    template <>
    struct Ops<std::complex<float>> {
        static force_inline __m256 unpack(std::complex<float> *arr) {
            return Ops<float>::unpack((float*)arr);
        }
        static force_inline __m256 unpack(std::complex<float> *arr0, std::complex<float> *arr1) {
            return Ops<float>::unpack((float*)arr0, (float*)arr1);
        }
        static force_inline __m256 unpack(std::complex<float> *arr0, std::complex<float> *arr1, std::complex<float> *arr2, std::complex<float> *arr3) {
            return Ops<float>::unpack((float*)arr0, (float*)arr1, (float*)arr2, (float*)arr3);
        }

        static force_inline void pack( __m256 src, std::complex<float> *arr) {
            Ops<float>::pack(src, (float*)arr);
        }        
        static force_inline void pack( __m256 src, std::complex<float> *arr0, std::complex<float> *arr1) {
            Ops<float>::pack(src, (float*)arr0, (float*)arr1);
        }        
        static force_inline void pack( __m256 src, std::complex<float> *arr0, std::complex<float> *arr1, std::complex<float> *arr2, std::complex<float> *arr3) {
            Ops<float>::pack(src, (float*)arr0, (float*)arr1, (float*)arr2, (float*)arr3);
        }        
    };
    
    template <class T, std::size_t index, std::size_t chunks>
    struct unpack;

    template <class T>
    struct unpack<T, 0, 1> {
        constexpr static std::size_t vals_per_chunk = 32 / sizeof(T);
        
        static force_inline void doIt(typename Treg<T>::val *vals, T **arr) {
            vals[0] = Ops<T>::unpack(arr[0] + 0 * vals_per_chunk);
        }
    };

    template <class T, std::size_t index>
    struct unpack<T, index, 1> {
        constexpr static std::size_t vals_per_chunk = 32 / sizeof(T);
        
        static force_inline void doIt(typename Treg<T>::val *vals, T **arr) {
            vals[index] = Ops<T>::unpack(arr[0] + index * vals_per_chunk);
            unpack<T, index - 1, 1>::doIt(vals, arr);
        }
    };

    template <class T>
    struct unpack<T, 0, 2> {
        static force_inline void doIt(typename Treg<T>::val *vals, T **arr) {
            vals[0] = Ops<T>::unpack(arr[0], arr[1]);
        }
    };

    template <class T>
    struct unpack<T, 0, 4> {
        static force_inline void doIt(typename Treg<T>::val *vals, T **arr) {
            vals[0] = Ops<T>::unpack(arr[0], arr[1], arr[2], arr[3]);
        }
    };
    
    template <class T>
    struct unpack<T, 0, 8> {
        static force_inline void doIt(typename Treg<T>::val *vals, T **arr) {
            vals[0] = Ops<T>::unpack(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]);
        }
    };
    
    template <class T, std::size_t index, std::size_t step>
    struct pack;

    template <class T>
    struct pack<T, 0, 1> {
        constexpr static std::size_t vals_per_chunk = 32 / sizeof(T);
        
        static force_inline void doIt(T **dst, typename Treg<T>::val *acc) {
            Ops<T>::pack(acc[0], dst[0] + 0 * vals_per_chunk);
        }
    };

    template <class T, std::size_t index>
    struct pack<T, index, 1> {
        constexpr static std::size_t vals_per_chunk = 32 / sizeof(T);
        
        static force_inline void doIt(T **dst, typename Treg<T>::val *acc) {
            Ops<T>::pack(acc[index], dst[0] + index * vals_per_chunk);
            pack<T, index - 1, 1>::doIt(dst, acc);
        }
    };

    template <class T>
    struct pack<T, 0, 2> {
        static force_inline void doIt(T **dst, typename Treg<T>::val *acc) {
            Ops<T>::pack(acc[0], dst[0], dst[1]);
        }
    };
    
    template <class T>
    struct pack<T, 0, 4> {
        static force_inline void doIt(T **dst, typename Treg<T>::val *acc) {
            Ops<T>::pack(acc[0], dst[0], dst[1], dst[2], dst[3]);
        }
    };
    
    template <class T>
    struct pack<T, 0, 8> {
        static force_inline void doIt(T **dst, typename Treg<T>::val *acc) {
            Ops<T>::pack(acc[0], dst[0], dst[1], dst[2], dst[3], dst[4], dst[5], dst[6], dst[7]);
        }
    };

    
    
    template <class T, std::size_t index>
    struct summation;
    
    template <>
    struct summation<std::complex<double>, 0> {
        static force_inline void doIt(__m256d *acc, __m256d *vals) {
            acc[0] = _mm256_add_pd(acc[0], vals[0]);
        }
    };

    template <std::size_t index>
    struct summation<std::complex<double>, index> {
        static force_inline void doIt(__m256d *acc, __m256d *vals) {
            acc[index] = _mm256_add_pd(acc[index], vals[index]);

            summation<std::complex<double>, index - 1>::doIt(acc, vals);
        }
    };
    
    template <>
    struct summation<double, 0> {
        static force_inline void doIt(__m256d *acc, __m256d *vals) {
            acc[0] = _mm256_add_pd(acc[0], vals[0]);
        }
    };
    
    template <std::size_t index>
    struct summation<double, index> {
        static force_inline void doIt(__m256d *acc, __m256d *vals) {
            acc[index] = _mm256_add_pd(acc[index], vals[index]);

            summation<double, index - 1>::doIt(acc, vals);
        }
    };
    
    template <>
    struct summation<std::complex<float>, 0> {
        static force_inline void doIt(__m256 *acc, __m256 *vals) {
            acc[0] = _mm256_add_ps(acc[0], vals[0]);
        }
    };

    template <std::size_t index>
    struct summation<std::complex<float>, index> {
        static force_inline void doIt(__m256 *acc, __m256 *vals) {
            acc[index] = _mm256_add_ps(acc[index], vals[index]);

            summation<std::complex<float>, index - 1>::doIt(acc, vals);
        }
    };
    
    template <>
    struct summation<float, 0> {
        static force_inline void doIt(__m256 *acc, __m256 *vals) {
            acc[0] = _mm256_add_ps(acc[0], vals[0]);
        }
    };
    
    template <std::size_t index>
    struct summation<float, index> {
        static force_inline void doIt(__m256 *acc, __m256 *vals) {
            acc[index] = _mm256_add_ps(acc[index], vals[index]);

            summation<float, index - 1>::doIt(acc, vals);
        }
    };
}
}


