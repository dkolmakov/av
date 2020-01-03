#pragma once

#include <immintrin.h>
#include <complex>

#include "common.hpp"

namespace mul_man {
    
namespace sse {

    template <class T, std::size_t index>
    struct unpack_complex;

    template <class T>
    struct unpack_complex<T, 0> {
        static force_inline void unpack(__m128d *realB, __m128d *imagB, std::complex<T> *arr) {
            __m128d B0 = _mm_loadu_pd((double*)(arr + 0 * 2));
            __m128d B2 = _mm_loadu_pd((double*)(arr + 0 * 2 + 1));
            realB[0] = _mm_unpacklo_pd(B0, B2);
            imagB[0] = _mm_unpackhi_pd(B0, B2);
        }
    };

    template <class T, std::size_t index>
    struct unpack_complex {
        static force_inline void unpack(__m128d *realB, __m128d *imagB, std::complex<T> *arr) {
            __m128d B0 = _mm_loadu_pd((double*)(arr + index * 2));
            __m128d B2 = _mm_loadu_pd((double*)(arr + index * 2 + 1));
            realB[index] = _mm_unpacklo_pd(B0, B2);
            imagB[index] = _mm_unpackhi_pd(B0, B2);
            unpack_complex<T, index - 1>::unpack(realB, imagB, arr);
        }
    };
    
    template <class T, std::size_t index>
    struct multiply;

    template <class T>
    struct multiply<T, 0> {
        static force_inline void compute(__m128d *realA, __m128d *imagA, __m128d *realB, __m128d *imagB) {
            __m128d realprod = _mm_mul_pd(realA[0], realB[0]);
            __m128d imagprod = _mm_mul_pd(imagA[0], imagB[0]);
            
            __m128d rAiB     = _mm_mul_pd(realA[0], imagB[0]);
            __m128d rBiA     = _mm_mul_pd(realB[0], imagA[0]);

            realA[0]     = _mm_sub_pd(realprod, imagprod);
            imagA[0]     = _mm_add_pd(rAiB, rBiA);
        }
    };

    template <class T, std::size_t index>
    struct multiply {
        static force_inline void compute(__m128d *realA, __m128d *imagA, __m128d *realB, __m128d *imagB) {
            __m128d realprod = _mm_mul_pd(realA[index], realB[index]);
            __m128d imagprod = _mm_mul_pd(imagA[index], imagB[index]);
            
            __m128d rAiB     = _mm_mul_pd(realA[index], imagB[index]);
            __m128d rBiA     = _mm_mul_pd(realB[index], imagA[index]);

            realA[index]     = _mm_sub_pd(realprod, imagprod);
            imagA[index]     = _mm_add_pd(rAiB, rBiA);

            multiply<T, index - 1>::compute(realA, imagA, realB, imagB);
        }
    };

    template <class T, std::size_t index>
    struct pack;

    template <class T>
    struct pack<T, 0> {
        static force_inline void make(__m128d *realA, __m128d *imagA, std::complex<T> *acc) {
            __m128d dst0 = _mm_shuffle_pd(realA[0], imagA[0], 0b00);
            __m128d dst2 = _mm_shuffle_pd(realA[0], imagA[0], 0b11);
            _mm_storeu_pd((double*)(acc + 0 * 2), dst0);
            _mm_storeu_pd((double*)(acc + 0 * 2 + 1), dst2);
        }
    };

    template <class T, std::size_t index>
    struct pack {
        static force_inline void make(__m128d *realA, __m128d *imagA, std::complex<T> *acc) {
            __m128d dst0 = _mm_shuffle_pd(realA[index], imagA[index], 0b00);
            __m128d dst2 = _mm_shuffle_pd(realA[index], imagA[index], 0b11);
            _mm_storeu_pd((double*)(acc + index * 2), dst0);
            _mm_storeu_pd((double*)(acc + index * 2 + 1), dst2);

            pack<T, index - 1>::make(realA, imagA, acc);
        }
    };
    
}


}


