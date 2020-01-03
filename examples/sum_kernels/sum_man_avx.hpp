#pragma once

#include <immintrin.h>
#include <complex>

#include "common.hpp"

namespace sum_man_avx {
    
namespace implementation {

    template <class T, std::size_t index>
    struct unpack;

    template <class T>
    struct unpack<T, 0> {
        static force_inline void doIt(__m256d *vals, std::complex<T> *arr) {
            vals[0] = _mm256_loadu_pd((double*)(arr + 0 * 2));
        }
    };

    template <class T, std::size_t index>
    struct unpack {
        static force_inline void doIt(__m256d *vals, std::complex<T> *arr) {
            vals[index] = _mm256_loadu_pd((double*)(arr + index * 2));
            unpack<T, index - 1>::doIt(vals, arr);
        }
    };

    template <class T, std::size_t index>
    struct summation;
    
    template <class T>
    struct summation<T, 0> {
        static force_inline void doIt(__m256d *acc, __m256d *vals) {
            acc[0] = _mm256_add_pd(acc[0], vals[0]);
        }
    };
    
    template <class T, std::size_t index>
    struct summation {
        static force_inline void doIt(__m256d *acc, __m256d *vals) {
            acc[index] = _mm256_add_pd(acc[index], vals[index]);

            summation<T, index - 1>::doIt(acc, vals);
        }
    };
    
    template <class T, std::size_t index>
    struct pack;

    template <class T>
    struct pack<T, 0> {
        static force_inline void doIt(std::complex<T> *dst, __m256d *acc) {
            _mm256_storeu_pd((double*)(dst + 0 * 2), acc[0]);
        }
    };

    template <class T, std::size_t index>
    struct pack {
        static force_inline void doIt(std::complex<T> *dst, __m256d *acc) {
            _mm256_storeu_pd((double*)(dst + index * 2), acc[index]);
            pack<T, index - 1>::doIt(dst, acc);
        }
    };

    
    template <class T, std::size_t chunk_size, std::size_t parity_checker, bool supported = av::SIMD_REG_SIZE >= 32 >
    struct chunk_sum;
    
    template <class T, std::size_t chunk_size>
    struct chunk_sum<T, chunk_size, 0, true> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr) {
            __m256d dataA[chunk_size];
            unpack<T, chunk_size - 1>::doIt(dataA, acc);
            __m256d dataB[chunk_size];
            unpack<T, chunk_size - 1>::doIt(dataB, arr);
            
            summation<T, chunk_size - 1>::doIt(dataA, dataB);

            pack<T, chunk_size - 1>::doIt(acc, dataA);
        }
    };

    template <class T, std::size_t chunk_size, std::size_t, bool>
    struct chunk_sum {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr) {
            for (std::size_t i = 0; i < chunk_size; i++) {
                acc[i] += arr[i];
            }
        }
    };

}

    struct chunk_sum {
        static std::string get_label() {
            return "sum_man_avx";
        }
        
        template <class T, std::size_t chunk_size>
        struct core {
            static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr) {
                implementation::chunk_sum<T, chunk_size, chunk_size % 2>::compute(acc, arr);
            }
        };
    };
   
}


