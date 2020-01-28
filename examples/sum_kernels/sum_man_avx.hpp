#pragma once

#include <immintrin.h>
#include <complex>

#include "common.hpp"

namespace sum_man {
    
namespace avx {    
    template <class Treg>
    struct Ops;
    
    template <>
    struct Ops<__m256d> {
        static force_inline __m256d unpack(void *arr) {
            double *to_load = (double *)arr;
            return _mm256_loadu_pd(to_load);
        }
        static force_inline __m256d unpack(void *arr0, void *arr1) {
            double *to_load0 = (double *)arr0;
            double *to_load1 = (double *)arr1;
            return _mm256_setr_pd(to_load0[0], to_load0[1], to_load1[0], to_load1[1]);
        }
        static force_inline __m256d unpack(void *arr0, void *arr1, void *arr2, void *arr3) {
            double *to_load0 = (double *)arr0;
            double *to_load1 = (double *)arr1;
            double *to_load2 = (double *)arr2;
            double *to_load3 = (double *)arr3;
            return _mm256_setr_pd(to_load0[0], to_load1[0], to_load2[0], to_load3[0]);
        }

        static force_inline void pack( __m256d src, void *arr) {
            double *to_store = (double *)arr;
            _mm256_storeu_pd(to_store, src);
        }        
        static force_inline void pack( __m256d src, void *arr0, void *arr1) {
            double *to_store0 = (double *)arr0;
            double *to_store1 = (double *)arr1;
            double* result = (double*)&src;
            to_store0[0] = result[0];
            to_store0[1] = result[1];
            to_store1[0] = result[2];
            to_store1[1] = result[3];
        }        
        static force_inline void pack( __m256d src, void *arr0, void *arr1, void *arr2, void *arr3) {
            double *to_store0 = (double *)arr0;
            double *to_store1 = (double *)arr1;
            double *to_store2 = (double *)arr2;
            double *to_store3 = (double *)arr3;
            double* result = (double*)&src;
            to_store0[0] = result[0];
            to_store1[1] = result[1];
            to_store2[2] = result[2];
            to_store3[3] = result[3];
        }        
    };
    
    template <>
    struct Ops<__m256> {
        static force_inline __m256 unpack(void *arr) {
            float *to_load = (float *)arr;
            return _mm256_loadu_ps(to_load);
        }
        static force_inline __m256 unpack(void *arr0, void *arr1) {
            float *to_load0 = (float *)arr0;
            float *to_load1 = (float *)arr1;
            return _mm256_setr_ps(to_load0[0], to_load0[1], to_load0[2], to_load0[3], to_load1[0], to_load1[1], to_load1[2], to_load1[3]);
        }
        static force_inline __m256 unpack(void *arr0, void *arr1, void *arr2, void *arr3) {
            float *to_load0 = (float *)arr0;
            float *to_load1 = (float *)arr1;
            float *to_load2 = (float *)arr2;
            float *to_load3 = (float *)arr3;
            return _mm256_setr_ps(to_load0[0], to_load0[1], to_load1[0], to_load1[1], to_load2[0], to_load2[1], to_load3[0], to_load3[1]);
        }
        static force_inline __m256 unpack(void *arr0, void *arr1, void *arr2, void *arr3, void *arr4, void *arr5, void *arr6, void *arr7) {
            float *to_load0 = (float *)arr0;
            float *to_load1 = (float *)arr1;
            float *to_load2 = (float *)arr2;
            float *to_load3 = (float *)arr3;
            float *to_load4 = (float *)arr4;
            float *to_load5 = (float *)arr5;
            float *to_load6 = (float *)arr6;
            float *to_load7 = (float *)arr7;
            return _mm256_setr_ps(to_load0[0], to_load1[0], to_load2[0], to_load3[0], to_load4[0], to_load5[0], to_load6[0], to_load7[0]);
        }

        static force_inline void pack( __m256 src, void *arr) {
            float *to_store = (float *)arr;
            _mm256_storeu_ps(to_store, src);
        }        
        static force_inline void pack( __m256 src, void *arr0, void *arr1) {
            float *to_store0 = (float *)arr0;
            float *to_store1 = (float *)arr1;
            float* result = (float*)&src;
            to_store0[0] = result[0]; to_store0[1] = result[1]; to_store0[2] = result[2]; to_store0[3] = result[3];
            to_store1[0] = result[4]; to_store1[1] = result[5]; to_store1[2] = result[6]; to_store1[3] = result[7];
        }        
        static force_inline void pack( __m256 src, void *arr0, void *arr1, void *arr2, void *arr3) {
            float *to_store0 = (float *)arr0;
            float *to_store1 = (float *)arr1;
            float *to_store2 = (float *)arr2;
            float *to_store3 = (float *)arr3;
            double* result = (double*)&src;
            to_store0[0] = result[0]; to_store0[1] = result[1];
            to_store1[0] = result[2]; to_store1[1] = result[3];
            to_store2[0] = result[4]; to_store2[1] = result[5];
            to_store3[0] = result[6]; to_store3[1] = result[7];
        }        
        static force_inline void pack( __m256 src, void *arr0, void *arr1, void *arr2, void *arr3, void *arr4, void *arr5, void *arr6, void *arr7) {
            float *to_store0 = (float *)arr0;
            float *to_store1 = (float *)arr1;
            float *to_store2 = (float *)arr2;
            float *to_store3 = (float *)arr3;
            float *to_store4 = (float *)arr4;
            float *to_store5 = (float *)arr5;
            float *to_store6 = (float *)arr6;
            float *to_store7 = (float *)arr7;
            double* result = (double*)&src;
            to_store0[0] = result[0];
            to_store1[0] = result[1];
            to_store2[0] = result[2];
            to_store3[0] = result[3];
            to_store4[0] = result[4];
            to_store5[0] = result[5];
            to_store6[0] = result[6];
            to_store7[0] = result[7];
        }        
    };

    
    template <class T, class Treg, std::size_t index, std::size_t chunks>
    struct unpack;

    template <class T, class Treg>
    struct unpack<T, Treg, 0, 1> {
        constexpr static std::size_t vals_per_chunk = 32 / sizeof(T);
        
        static force_inline void doIt(Treg *vals, T **arr) {
            vals[0] = Ops<Treg>::unpack(arr[0] + 0 * vals_per_chunk);
        }
    };

    template <class T, class Treg, std::size_t index>
    struct unpack<T, Treg, index, 1> {
        constexpr static std::size_t vals_per_chunk = 32 / sizeof(T);
        
        static force_inline void doIt(Treg *vals, T **arr) {
            vals[index] = Ops<Treg>::unpack(arr[0] + index * vals_per_chunk);
            unpack<T, Treg, index - 1, 1>::doIt(vals, arr);
        }
    };

    template <class T, class Treg>
    struct unpack<T, Treg, 0, 2> {
        static force_inline void doIt(Treg *vals, T **arr) {
            vals[0] = Ops<Treg>::unpack(arr[0], arr[1]);
        }
    };

    template <class T, class Treg>
    struct unpack<T, Treg, 0, 4> {
        static force_inline void doIt(Treg *vals, T **arr) {
            vals[0] = Ops<Treg>::unpack(arr[0], arr[1], arr[2], arr[3]);
        }
    };
    
    template <class T, class Treg>
    struct unpack<T, Treg, 0, 8> {
        static force_inline void doIt(Treg *vals, T **arr) {
            vals[0] = Ops<Treg>::unpack(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]);
        }
    };
    
    template <class T, class Treg, std::size_t index, std::size_t step>
    struct pack;

    template <class T, class Treg>
    struct pack<T, Treg, 0, 1> {
        constexpr static std::size_t vals_per_chunk = 32 / sizeof(T);
        
        static force_inline void doIt(T **dst, Treg *acc) {
            Ops<Treg>::pack(acc[0], dst[0] + 0 * vals_per_chunk);
        }
    };

    template <class T, class Treg, std::size_t index>
    struct pack<T, Treg, index, 1> {
        constexpr static std::size_t vals_per_chunk = 32 / sizeof(T);
        
        static force_inline void doIt(T **dst, Treg *acc) {
            Ops<Treg>::pack(acc[index], dst[0] + index * vals_per_chunk);
            pack<T, Treg, index - 1, 1>::doIt(dst, acc);
        }
    };

    template <class T, class Treg>
    struct pack<T, Treg, 0, 2> {
        static force_inline void doIt(T **dst, Treg *acc) {
            Ops<Treg>::pack(acc[0], dst[0], dst[1]);
        }
    };
    
    template <class T, class Treg>
    struct pack<T, Treg, 0, 4> {
        static force_inline void doIt(T **dst, Treg *acc) {
            Ops<Treg>::pack(acc[0], dst[0], dst[1], dst[2], dst[3]);
        }
    };
    
    template <class T, class Treg>
    struct pack<T, Treg, 0, 8> {
        static force_inline void doIt(T **dst, Treg *acc) {
            Ops<Treg>::pack(acc[0], dst[0], dst[1], dst[2], dst[3], dst[4], dst[5], dst[6], dst[7]);
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


