#pragma once

#include <immintrin.h>
#include <complex>

#include "common.hpp"

namespace av_mul_avx {
    
namespace implementation {

    // Initial idea took from https://stackoverflow.com/questions/39509746/how-to-square-two-complex-doubles-with-256-bit-avx-vectors
    
    template <class T, std::size_t chunk_size>
    struct init_real;

    template <class T>
    struct init_real<T, 0> {
        static force_inline void init(__m256d *real) {
            real[0] = _mm256_setr_pd(1.0, 1.0, 1.0, 1.0);
        }
    };

    template <class T, std::size_t chunk_size>
    struct init_real {
        static force_inline void init(__m256d *real) {
            real[chunk_size] = _mm256_setr_pd(1.0, 1.0, 1.0, 1.0);
            init_real<T, chunk_size - 1>::init(real);
        }
    };


    template <class T, std::size_t chunk_size>
    struct init_imag;

    template <class T>
    struct init_imag<T, 0> {
        static force_inline void init(__m256d *imag) {
            imag[0] = _mm256_setr_pd(0, 0, 0, 0);
        }
    };

    template <class T, std::size_t chunk_size>
    struct init_imag {
        static force_inline void init(__m256d *imag) {
            imag[chunk_size] = _mm256_setr_pd(0, 0, 0, 0);
            init_imag<T, chunk_size - 1>::init(imag);
        }
    };


    template <class T, std::size_t index>
    struct unpackB;

    template <class T>
    struct unpackB<T, 0> {
        static force_inline void unpack(__m256d *realB, __m256d *imagB, std::complex<T> *arr) {
            __m256d B0 = _mm256_loadu_pd((double*)(arr + 0 * 4));
            __m256d B2 = _mm256_loadu_pd((double*)(arr + 0 * 4 + 2));
            realB[0] = _mm256_unpacklo_pd(B0, B2);
            imagB[0] = _mm256_unpackhi_pd(B0, B2);
        }
    };

    template <class T, std::size_t index>
    struct unpackB {
        static force_inline void unpack(__m256d *realB, __m256d *imagB, std::complex<T> *arr) {
            __m256d B0 = _mm256_loadu_pd((double*)(arr + index * 4));
            __m256d B2 = _mm256_loadu_pd((double*)(arr + index * 4 + 2));
            realB[index] = _mm256_unpacklo_pd(B0, B2);
            imagB[index] = _mm256_unpackhi_pd(B0, B2);
            unpackB<T, index - 1>::unpack(realB, imagB, arr);
        }
    };
    
    template <class T, std::size_t index>
    struct multiply;

    template <class T>
    struct multiply<T, 0> {
        static force_inline void compute(__m256d *realA, __m256d *imagA, __m256d *realB, __m256d *imagB) {
            __m256d realprod = _mm256_mul_pd(realA[0], realB[0]);
            __m256d imagprod = _mm256_mul_pd(imagA[0], imagB[0]);
            
            __m256d rAiB     = _mm256_mul_pd(realA[0], imagB[0]);
            __m256d rBiA     = _mm256_mul_pd(realB[0], imagA[0]);

            realA[0]     = _mm256_sub_pd(realprod, imagprod);
            imagA[0]     = _mm256_add_pd(rAiB, rBiA);
        }
    };

    template <class T, std::size_t index>
    struct multiply {
        static force_inline void compute(__m256d *realA, __m256d *imagA, __m256d *realB, __m256d *imagB) {
            __m256d realprod = _mm256_mul_pd(realA[index], realB[index]);
            __m256d imagprod = _mm256_mul_pd(imagA[index], imagB[index]);
            
            __m256d rAiB     = _mm256_mul_pd(realA[index], imagB[index]);
            __m256d rBiA     = _mm256_mul_pd(realB[index], imagA[index]);

            realA[index]     = _mm256_sub_pd(realprod, imagprod);
            imagA[index]     = _mm256_add_pd(rAiB, rBiA);

            multiply<T, index - 1>::compute(realA, imagA, realB, imagB);
        }
    };
    
    template <class T, std::size_t index>
    struct pack;

    template <class T>
    struct pack<T, 0> {
        static force_inline void make(__m256d *realA, __m256d *imagA, std::complex<T> *acc) {
            __m256d dst0 = _mm256_shuffle_pd(realA[0], imagA[0], 0b0000);
            __m256d dst2 = _mm256_shuffle_pd(realA[0], imagA[0], 0b1111);
            _mm256_storeu_pd((double*)(acc + 0 * 4), dst0);
            _mm256_storeu_pd((double*)(acc + 0 * 4 + 2), dst2);
        }
    };

    template <class T, std::size_t index>
    struct pack {
        static force_inline void make(__m256d *realA, __m256d *imagA, std::complex<T> *acc) {
            __m256d dst0 = _mm256_shuffle_pd(realA[index], imagA[index], 0b0000);
            __m256d dst2 = _mm256_shuffle_pd(realA[index], imagA[index], 0b1111);
            _mm256_storeu_pd((double*)(acc + index * 4), dst0);
            _mm256_storeu_pd((double*)(acc + index * 4 + 2), dst2);

            pack<T, index - 1>::make(realA, imagA, acc);
        }
    };
    
    
    template <class T, std::size_t chunk_size, std::size_t parity_checker>
    struct chunk_mul;
    
    template <class T, std::size_t chunk_size>
    struct chunk_mul<T, chunk_size, 0> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr, std::size_t count) {
            __m256d realA[chunk_size / 4];
            init_real<T, chunk_size / 4 - 1>::init(realA);
            __m256d imagA[chunk_size / 4]; 
            init_imag<T, chunk_size / 4 - 1>::init(imagA);
            
            for (std::size_t i = 0; i < count; i += chunk_size) {
                __m256d realB[chunk_size / 4];
                __m256d imagB[chunk_size / 4];
                
                unpackB<T, chunk_size / 4 - 1>::unpack(realB, imagB, arr);
                multiply<T, chunk_size / 4 - 1>::compute(realA, imagA, realB, imagB);
            }
            
            pack<T, chunk_size / 4 - 1>::make(realA, imagA, acc);
        }
    };
    
    template <class T, std::size_t chunk_size, std::size_t parity_checker>
    struct chunk_mul {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr, std::size_t count) {
            // Default implementation
            for (std::size_t i = 0; i < chunk_size; i++)
                acc[i] = 1;
            
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
    struct mul<T, chunk_size, 32> {
        static force_inline std::complex<T> compute(std::complex<T> *arr, const std::size_t count) {
            // Specialized implementation
            std::complex<T> acc[chunk_size];
            std::size_t to_mul = count - count % chunk_size;
            
            asm volatile ("nop;nop;nop;");
            chunk_mul<T, chunk_size, chunk_size % 4>::compute(acc, arr, to_mul);
            asm volatile ("nop;nop;nop;");

            std::size_t i = to_mul;
            
            // Handle the remainder
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
            chunk_mul<T, 1, 1>::compute(&acc, arr, count);
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


