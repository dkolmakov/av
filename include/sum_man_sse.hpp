#pragma once

#include <immintrin.h>
#include <complex>

#include "common.hpp"

namespace sum_man_sse {
    
namespace implementation {

    template <class T, std::size_t index>
    struct unpack;

    template <class T>
    struct unpack<T, 0> {
        static force_inline void doIt(__m128d *vals, std::complex<T> *arr) {
            vals[0] = _mm_loadu_pd((double*)(arr + 0));
        }
    };

    template <class T, std::size_t index>
    struct unpack {
        static force_inline void doIt(__m128d *vals, std::complex<T> *arr) {
            vals[index] = _mm_loadu_pd((double*)(arr + index));
            unpack<T, index - 1>::doIt(vals, arr);
        }
    };

    template <class T, std::size_t index>
    struct summation;
    
    template <class T>
    struct summation<T, 0> {
        static force_inline void doIt(__m128d *acc, __m128d *vals) {
            acc[0] = _mm_add_pd(acc[0], vals[0]);
        }
    };
    
    template <class T, std::size_t index>
    struct summation {
        static force_inline void doIt(__m128d *acc, __m128d *vals) {
            acc[index] = _mm_add_pd(acc[index], vals[index]);

            summation<T, index - 1>::doIt(acc, vals);
        }
    };
    
    template <class T, std::size_t index>
    struct pack;

    template <class T>
    struct pack<T, 0> {
        static force_inline void doIt(std::complex<T> *dst, __m128d *acc) {
            _mm_storeu_pd((double*)(dst +  0), acc[0]);
        }
    };

    template <class T, std::size_t index>
    struct pack {
        static force_inline void doIt(std::complex<T> *dst, __m128d *acc) {
            _mm_storeu_pd((double*)(dst + index), acc[index]);
            pack<T, index - 1>::doIt(dst, acc);
        }
    };

    template <class T, std::size_t chunk_size, bool supported = av::SIMD_REG_SIZE >= 16 >
    struct chunk_sum;
    
    template <class T, std::size_t chunk_size>
    struct chunk_sum<T, chunk_size, true> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr) {
            __m128d dataA[chunk_size];
            unpack<T, chunk_size - 1>::doIt(dataA, acc);
            __m128d dataB[chunk_size];
            unpack<T, chunk_size - 1>::doIt(dataB, arr);
            
            summation<T, chunk_size - 1>::doIt(dataA, dataB);

            pack<T, chunk_size - 1>::doIt(acc, dataA);
        }
    };
    
    template <class T, std::size_t chunk_size, bool>
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
            return "sum_man_sse";
        }
        
        template <class T, std::size_t chunk_size>
        struct core {
            static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr) {
                implementation::chunk_sum<T, chunk_size>::compute(acc, arr);
            }
        };
    };
    
}


