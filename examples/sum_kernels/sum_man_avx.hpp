#pragma once

#include <immintrin.h>
#include <complex>

#include "common.hpp"

namespace sum_man {
    
namespace avx {

    template <class T, std::size_t index, std::size_t chunks>
    struct unpack;

    template <>
    struct unpack<std::complex<double>, 0, 1> {
        constexpr static std::size_t vals_per_chunk = 32 / sizeof(std::complex<double>);
        
        static force_inline void doIt(__m256d *vals, std::complex<double> **arr) {
            vals[0] = _mm256_loadu_pd((double*)(arr[0] + 0 * vals_per_chunk));
        }
    };

    template <std::size_t index>
    struct unpack<std::complex<double>, index, 1> {
        constexpr static std::size_t vals_per_chunk = 32 / sizeof(std::complex<double>);
        
        static force_inline void doIt(__m256d *vals, std::complex<double> **arr) {
            vals[index] = _mm256_loadu_pd((double*)(arr[0] + index * vals_per_chunk));
            unpack<std::complex<double>, index - 1, 1>::doIt(vals, arr);
        }
    };

    template <>
    struct unpack<std::complex<double>, 0, 2> {
        static force_inline void doIt(__m256d *vals, std::complex<double> **arr) {
            double *to_load0 = (double*)arr[0];
            double *to_load1 = (double*)arr[1];
            vals[0] = _mm256_setr_pd(to_load0[0], to_load0[1], to_load1[0], to_load1[1]);
        }
    };

    
    
    template <>
    struct unpack<std::complex<float>, 0, 1> {
        constexpr static std::size_t vals_per_chunk = 32 / sizeof(std::complex<float>);
        
        static force_inline void doIt(__m256 *vals, std::complex<float> **arr) {
            vals[0] = _mm256_loadu_ps((float*)(arr[0] + 0 * vals_per_chunk));
        }
    };

    template <std::size_t index>
    struct unpack<std::complex<float>, index, 1> {
        constexpr static std::size_t vals_per_chunk = 32 / sizeof(std::complex<float>);
        
        static force_inline void doIt(__m256 *vals, std::complex<float> **arr) {
            vals[index] = _mm256_loadu_ps((float*)(arr[0] + index * vals_per_chunk));
            unpack<std::complex<float>, index - 1, 1>::doIt(vals, arr);
        }
    };

    template <>
    struct unpack<std::complex<float>, 0, 2> {
        static force_inline void doIt(__m256 *vals, std::complex<float> **arr) {
            float *to_load0 = (float*)arr[0];
            float *to_load1 = (float*)arr[1];
            vals[0] = _mm256_setr_ps(to_load0[0], to_load0[1], to_load0[2], to_load0[3], to_load1[0], to_load1[1], to_load1[2], to_load1[3]);
        }
    };
    
    template <>
    struct unpack<std::complex<float>, 0, 4> {
        static force_inline void doIt(__m256 *vals, std::complex<float> **arr) {
            float *to_load0 = (float*)arr[0];
            float *to_load1 = (float*)arr[1];
            float *to_load2 = (float*)arr[2];
            float *to_load3 = (float*)arr[3];
            vals[0] = _mm256_setr_ps(to_load0[0], to_load0[1], to_load1[0], to_load2[1], to_load2[0], to_load2[1], to_load3[0], to_load3[1]);
        }
    };
    


    template <class T, std::size_t index, std::size_t step>
    struct pack;

    template <>
    struct pack<std::complex<double>, 0, 1> {
        constexpr static std::size_t vals_per_chunk = 32 / sizeof(std::complex<double>);
        
        static force_inline void doIt(std::complex<double> **dst, __m256d *acc) {
            _mm256_storeu_pd((double*)(dst[0] + 0 * vals_per_chunk), acc[0]);
        }
    };

    template <std::size_t index>
    struct pack<std::complex<double>, index, 1> {
        constexpr static std::size_t vals_per_chunk = 32 / sizeof(std::complex<double>);
        
        static force_inline void doIt(std::complex<double> **dst, __m256d *acc) {
            _mm256_storeu_pd((double*)(dst[0] + index * vals_per_chunk), acc[index]);
            pack<std::complex<double>, index - 1, 1>::doIt(dst, acc);
        }
    };

    template <>
    struct pack<std::complex<double>, 0, 2> {
        static force_inline void doIt(std::complex<double> **dst, __m256d *acc) {
            double *to_store0 = (double *)dst[0];
            double *to_store1 = (double *)dst[1];
            double* result = (double*)acc;
            to_store0[0] = result[0];
            to_store0[1] = result[1];
            to_store1[0] = result[2];
            to_store1[1] = result[3];
        }
    };
    
    
    
    template <>
    struct pack<std::complex<float>, 0, 1> {
        constexpr static std::size_t vals_per_chunk = 32 / sizeof(std::complex<float>);
        
        static force_inline void doIt(std::complex<float> **dst, __m256 *acc) {
            _mm256_storeu_ps((float*)(dst[0] + 0 * vals_per_chunk), acc[0]);
        }
    };

    template <std::size_t index>
    struct pack<std::complex<float>, index, 1> {
        constexpr static std::size_t vals_per_chunk = 32 / sizeof(std::complex<float>);
        
        static force_inline void doIt(std::complex<float> **dst, __m256 *acc) {
            _mm256_storeu_ps((float*)(dst[0] + index * vals_per_chunk), acc[index]);
            pack<std::complex<float>, index - 1, 1>::doIt(dst, acc);
        }
    };

    template <>
    struct pack<std::complex<float>, 0, 2> {
        static force_inline void doIt(std::complex<float> **dst, __m256 *acc) {
            float *to_store0 = (float *)dst[0];
            float *to_store1 = (float *)dst[1];
            float* result = (float*)acc;
            to_store0[0] = result[0];
            to_store0[1] = result[1];
            to_store0[2] = result[2];
            to_store0[3] = result[3];
            to_store1[0] = result[4];
            to_store1[1] = result[5];
            to_store1[2] = result[6];
            to_store1[3] = result[7];
        }
    };
    
    template <>
    struct pack<std::complex<float>, 0, 4> {
        static force_inline void doIt(std::complex<float> **dst, __m256 *acc) {
            float *to_store0 = (float *)dst[0];
            float *to_store1 = (float *)dst[1];
            float *to_store2 = (float *)dst[2];
            float *to_store3 = (float *)dst[3];
            float* result = (float*)acc;
            to_store0[0] = result[0];
            to_store0[1] = result[1];
            to_store1[0] = result[2];
            to_store1[1] = result[3];
            to_store2[0] = result[4];
            to_store2[1] = result[5];
            to_store3[0] = result[6];
            to_store3[1] = result[7];
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


