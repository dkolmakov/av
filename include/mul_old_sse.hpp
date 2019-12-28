#pragma once

#include <immintrin.h>
#include <complex>

#include "common.hpp"

namespace mul_old_sse {
    
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
    struct multiply;
    
    template <class T>
    struct multiply<T, 0> {
        static force_inline void doIt(__m128d *acc, __m128d *vals) {
            __m128d tmp0 = _mm_mul_pd(acc[0], vals[0]);

            vals[0] = _mm_shuffle_pd(vals[0], vals[0], 1);
            __m128d odd_signbits = _mm_setr_pd(0, -0.0);
            __m128d tmp1 = _mm_xor_pd(acc[0], odd_signbits);
            
            __m128d tmp2 = _mm_mul_pd(tmp1, vals[0]);
            
            acc[0] = _mm_addsub_pd(tmp0, tmp2);
        }
    };
    
    template <class T, std::size_t index>
    struct multiply {
        static force_inline void doIt(__m128d *acc, __m128d *vals) {
            __m128d tmp0 = _mm_mul_pd(acc[index], vals[index]);

            vals[index] = _mm_shuffle_pd(vals[index], vals[index], 1);
            __m128d odd_signbits = _mm_setr_pd(0, -0.0);
            acc[index] = _mm_xor_pd(acc[index], odd_signbits);
            
            __m128d tmp2 = _mm_mul_pd(acc[index], vals[index]);
            
            acc[index] = _mm_addsub_pd(tmp0, tmp2);

            multiply<T, index - 1>::doIt(acc, vals);
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
    
    // Original version id from https://www.codeproject.com/Articles/874396/Crunching-Numbers-with-AVX-and-AVX
    // And improved one is here https://stackoverflow.com/questions/39509746/how-to-square-two-complex-doubles-with-256-bit-avx-vectors

    template <class T, std::size_t chunk_size, std::size_t reg_size = av::SIMD_REG_SIZE>
    struct chunk_mul;
    
    template <class T, std::size_t chunk_size>
    struct chunk_mul<T, chunk_size, 16> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr) {
            __m128d res[chunk_size];
            unpack<T, chunk_size - 1>::doIt(res, acc);
            
            __m128d v0[chunk_size];
            unpack<T, chunk_size - 1>::doIt(v0, arr);
            multiply<T, chunk_size - 1>::doIt(res, v0);
            
            pack<T, chunk_size - 1>::doIt(acc, res);
        }
    };

    template <class T, std::size_t chunk_size, std::size_t reg_size>
    struct chunk_mul {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr) {
            // Default implementation
            for (std::size_t i = 0; i < chunk_size; i++)
                acc[i] *= arr[i];
        }
    };
}

    struct chunk_mul {
        static std::string get_label() {
            return "mul_old_sse";
        }
        
        template <class T, std::size_t chunk_size>
        struct core {
            static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr) {
                implementation::chunk_mul<T, chunk_size>::compute(acc, arr);
            }
        };
    };

}


