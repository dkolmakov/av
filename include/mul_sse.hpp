#pragma once

#include <immintrin.h>
#include <complex>

#include "common.hpp"

namespace av_mul_sse {
    
namespace implementation {

    template <class T, std::size_t chunk_size>
    struct init_real;

    template <class T>
    struct init_real<T, 0> {
        static force_inline void init(__m128d *real) {
            real[0] = _mm_setr_pd(1.0, 1.0);
        }
    };

    template <class T, std::size_t chunk_size>
    struct init_real {
        static force_inline void init(__m128d *real) {
            real[chunk_size] = _mm_setr_pd(1.0, 1.0);
            init_real<T, chunk_size - 1>::init(real);
        }
    };


    template <class T, std::size_t chunk_size>
    struct init_imag;

    template <class T>
    struct init_imag<T, 0> {
        static force_inline void init(__m128d *imag) {
            imag[0] = _mm_setr_pd(0, 0);
        }
    };

    template <class T, std::size_t chunk_size>
    struct init_imag {
        static force_inline void init(__m128d *imag) {
            imag[chunk_size] = _mm_setr_pd(0, 0);
            init_imag<T, chunk_size - 1>::init(imag);
        }
    };

    template <class T, std::size_t index>
    struct unpackB;

    template <class T>
    struct unpackB<T, 0> {
        static force_inline void unpack(__m128d *realB, __m128d *imagB, std::complex<T> *arr) {
            __m128d B0 = _mm_loadu_pd((double*)(arr + 0 * 2));
            __m128d B2 = _mm_loadu_pd((double*)(arr + 0 * 2 + 1));
            realB[0] = _mm_unpacklo_pd(B0, B2);
            imagB[0] = _mm_unpackhi_pd(B0, B2);
        }
    };

    template <class T, std::size_t index>
    struct unpackB {
        static force_inline void unpack(__m128d *realB, __m128d *imagB, std::complex<T> *arr) {
            __m128d B0 = _mm_loadu_pd((double*)(arr + index * 2));
            __m128d B2 = _mm_loadu_pd((double*)(arr + index * 2 + 1));
            realB[index] = _mm_unpacklo_pd(B0, B2);
            imagB[index] = _mm_unpackhi_pd(B0, B2);
            unpackB<T, index - 1>::unpack(realB, imagB, arr);
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
    
    template <class T, std::size_t chunk_size, std::size_t parity_checker>
    struct chunk_mul;
    
    template <class T, std::size_t chunk_size>
    struct chunk_mul<T, chunk_size, 0> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr, std::size_t count) {
            __m128d realA[chunk_size / 2];
            init_real<T, chunk_size / 2 - 1>::init(realA);
            __m128d imagA[chunk_size / 2]; 
            init_imag<T, chunk_size / 2 - 1>::init(imagA);
            
            for (std::size_t i = 0; i < count; i += chunk_size) {
                __m128d realB[chunk_size / 2];
                __m128d imagB[chunk_size / 2];
                
                unpackB<T, chunk_size / 2 - 1>::unpack(realB, imagB, arr);
                multiply<T, chunk_size / 2 - 1>::compute(realA, imagA, realB, imagB);
            }
            
            pack<T, chunk_size / 2 - 1>::make(realA, imagA, acc);
        }
    };
    
    template <class T, std::size_t chunk_size, std::size_t parity_checker>
    struct chunk_mul {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr, std::size_t count) {
            // Default implementation
            std::complex<T> res(1,0);
            for (std::size_t i = 0; i < count; i++) {
                res *= arr[i];
            }
            *acc = res;
        }
    };
    
    template <class T, std::size_t chunk_size, std::size_t reg_size = av::SIMD_REG_SIZE>
    struct mul;
    
    template <class T, std::size_t chunk_size>
    struct mul<T, chunk_size, 16> {
        static force_inline std::complex<T> compute(std::complex<T> *arr, const std::size_t count) {
            // Specialized implementation
            std::complex<T> acc[chunk_size];
            std::size_t to_mul = count - count % chunk_size;
            
            // Sum by chunks
            asm volatile ("nop;nop;nop;");
            chunk_mul<T, chunk_size, chunk_size % 2>::compute(acc, arr, to_mul);
            asm volatile ("nop;nop;nop;");

            std::size_t i = to_mul;
            
            // Add the remainder
            std::complex<T> result(1,0);
            std::size_t j = 0;
            for (; i < count; i++, j++) {
                result *= arr[i] * acc[j];
            }
            for (; j < chunk_size; j++) {
                result *= acc[j];
            }
            
            return result;
        }
    };
    
    template <class T, std::size_t chunk_size, std::size_t reg_size>
    struct mul {
        static force_inline std::complex<T> compute(std::complex<T> *arr, std::size_t count) {
            // Default implementation
            std::complex<T> acc(1,0);
            
            chunk_mul<T, chunk_size, 1>::compute(&acc, arr, count);
            
            return acc;
        }
    };
    
}

    template<class T, std::size_t chunk_size>
    static std::complex<T> mul(std::complex<T> *arr, std::size_t count) {
        return implementation::mul<T, chunk_size>::compute(arr, count);
    }

    template<class T, std::size_t chunk_size>
    struct ToTest {
        static std::complex<T> to_test(std::complex<T> *arr, std::size_t count) {
            return mul<T, chunk_size>(arr, count);
        }
    };

}


