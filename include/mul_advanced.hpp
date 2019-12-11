#pragma once

#include <immintrin.h>
#include <complex>

#include "common.hpp"

namespace av_mul_advanced {
    
namespace implementation {

    template <class T, std::size_t chunk_size>
    struct chunk_mul;

    template <class T>
    struct chunk_mul<T, 2> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr, std::size_t count) {
            __m128d realA = _mm_setr_pd(1.0, 1.0);
            __m128d imagA = _mm_setr_pd(0, 0);
            
            for (std::size_t i = 0; i < count; i += 2) {
                __m128d B0 = _mm_loadu_pd((double*)(arr + i));                       // [m n o p]
                __m128d B2 = _mm_loadu_pd((double*)(arr + i + 1));                   // [q r s t]
                __m128d realB = _mm_unpacklo_pd(B0, B2);                         // [m q o s]
                __m128d imagB = _mm_unpackhi_pd(B0, B2);                         // [n r p t]
                
                // desired:  real=rArB - iAiB,  imag=rAiB + rBiA
                __m128d realprod = _mm_mul_pd(realA, realB);
                __m128d imagprod = _mm_mul_pd(imagA, imagB);
                
                __m128d rAiB     = _mm_mul_pd(realA, imagB);
                __m128d rBiA     = _mm_mul_pd(realB, imagA);

                // gcc and clang will contract these into FMA.  (clang needs -ffp-contract=fast)
                // Doing it manually would remove the option to compile for non-FMA targets
                realA     = _mm_sub_pd(realprod, imagprod);  // [D0r D2r | D1r D3r]
                imagA     = _mm_add_pd(rAiB, rBiA);          // [D0i D2i | D1i D3i]                
            }
            
            // interleave the separate real and imaginary vectors back into packed format
            __m128d dst0 = _mm_shuffle_pd(realA, imagA, 0b00);  // [D0r D0i | D1r D1i]
            __m128d dst2 = _mm_shuffle_pd(realA, imagA, 0b11);  // [D2r D2i | D3r D3i]
            _mm_storeu_pd((double*) acc, dst0);
            _mm_storeu_pd((double*)(acc + 1), dst2);
        }
    };

    template <class T>
    struct chunk_mul<T, 4> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr, std::size_t count) {
            __m256d realA = _mm256_setr_pd(1.0, 1.0, 1.0, 1.0);
            __m256d imagA = _mm256_setr_pd(0, 0, 0, 0);
            
            for (std::size_t i = 0; i < count; i += 4) {
                // Took from https://stackoverflow.com/questions/39509746/how-to-square-two-complex-doubles-with-256-bit-avx-vectors
                __m256d B0 = _mm256_loadu_pd((double*)(arr + i));                       // [m n o p]
                __m256d B2 = _mm256_loadu_pd((double*)(arr + i + 2));                   // [q r s t]
                __m256d realB = _mm256_unpacklo_pd(B0, B2);                         // [m q o s]
                __m256d imagB = _mm256_unpackhi_pd(B0, B2);                         // [n r p t]

                // desired:  real=rArB - iAiB,  imag=rAiB + rBiA
                __m256d realprod = _mm256_mul_pd(realA, realB);
                __m256d imagprod = _mm256_mul_pd(imagA, imagB);
                
                __m256d rAiB     = _mm256_mul_pd(realA, imagB);
                __m256d rBiA     = _mm256_mul_pd(realB, imagA);

                // gcc and clang will contract these into FMA.  (clang needs -ffp-contract=fast)
                // Doing it manually would remove the option to compile for non-FMA targets
                realA     = _mm256_sub_pd(realprod, imagprod);  // [D0r D2r | D1r D3r]
                imagA     = _mm256_add_pd(rAiB, rBiA);          // [D0i D2i | D1i D3i]
            }
            
            // interleave the separate real and imaginary vectors back into packed format
            __m256d dst0 = _mm256_shuffle_pd(realA, imagA, 0b0000);  // [D0r D0i | D1r D1i]
            __m256d dst2 = _mm256_shuffle_pd(realA, imagA, 0b1111);  // [D2r D2i | D3r D3i]
            _mm256_storeu_pd((double*) acc, dst0);
            _mm256_storeu_pd((double*)(acc + 2), dst2);
        }
    };

    template <class T>
    struct chunk_mul<T, 8> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr, std::size_t count) {
            __m256d realA = _mm256_setr_pd(1.0, 1.0, 1.0, 1.0);
            __m256d imagA = _mm256_setr_pd(0, 0, 0, 0);
            __m256d realC = _mm256_setr_pd(1.0, 1.0, 1.0, 1.0);
            __m256d imagC = _mm256_setr_pd(0, 0, 0, 0);
            
            for (std::size_t i = 0; i < count; i += 8) {
                // Took from https://stackoverflow.com/questions/39509746/how-to-square-two-complex-doubles-with-256-bit-avx-vectors
                __m256d B0 = _mm256_loadu_pd((double*)(arr + i));                       // [m n o p]
                __m256d B2 = _mm256_loadu_pd((double*)(arr + i + 2));                   // [q r s t]
                __m256d B4 = _mm256_loadu_pd((double*)(arr + i + 4));                   // [m n o p]
                __m256d B6 = _mm256_loadu_pd((double*)(arr + i + 6));                   // [q r s t]
                __m256d realB = _mm256_unpacklo_pd(B0, B2);                         // [m q o s]
                __m256d imagB = _mm256_unpackhi_pd(B0, B2);                         // [n r p t]
                __m256d realD = _mm256_unpacklo_pd(B4, B6);                         // [m q o s]
                __m256d imagD = _mm256_unpackhi_pd(B4, B6);                         // [n r p t]

                // desired:  real=rArB - iAiB,  imag=rAiB + rBiA
                __m256d realprod0 = _mm256_mul_pd(realA, realB);
                __m256d imagprod0 = _mm256_mul_pd(imagA, imagB);
                __m256d realprod1 = _mm256_mul_pd(realC, realD);
                __m256d imagprod1 = _mm256_mul_pd(imagC, imagD);
                
                __m256d rAiB0     = _mm256_mul_pd(realA, imagB);
                __m256d rBiA0     = _mm256_mul_pd(realB, imagA);
                __m256d rAiB1     = _mm256_mul_pd(realC, imagD);
                __m256d rBiA1     = _mm256_mul_pd(realC, imagD);

                // gcc and clang will contract these into FMA.  (clang needs -ffp-contract=fast)
                // Doing it manually would remove the option to compile for non-FMA targets
                realA     = _mm256_sub_pd(realprod0, imagprod0);  // [D0r D2r | D1r D3r]
                imagA     = _mm256_add_pd(rAiB0, rBiA0);          // [D0i D2i | D1i D3i]
                realC     = _mm256_sub_pd(realprod1, imagprod1);  // [D0r D2r | D1r D3r]
                imagC     = _mm256_add_pd(rAiB1, rBiA1);          // [D0i D2i | D1i D3i]
            }
            
            // interleave the separate real and imaginary vectors back into packed format
            __m256d dst0 = _mm256_shuffle_pd(realA, imagA, 0b0000);  // [D0r D0i | D1r D1i]
            __m256d dst2 = _mm256_shuffle_pd(realA, imagA, 0b1111);  // [D2r D2i | D3r D3i]
            _mm256_storeu_pd((double*) acc, dst0);
            _mm256_storeu_pd((double*)(acc + 2), dst2);
            __m256d dst4 = _mm256_shuffle_pd(realC, imagC, 0b0000);  // [D0r D0i | D1r D1i]
            __m256d dst6 = _mm256_shuffle_pd(realC, imagC, 0b1111);  // [D2r D2i | D3r D3i]
            _mm256_storeu_pd((double*)(acc + 4), dst4);
            _mm256_storeu_pd((double*)(acc + 6), dst6);
        }
    };

    template <class T, std::size_t chunk_size>
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
            
            // Sum by chunks
            asm volatile ("nop;nop;nop;");
            chunk_mul<T, chunk_size>::compute(acc, arr, to_mul);
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
            
            chunk_mul<T, 0>::compute(&acc, arr, count);
            
            return acc;
        }
    };
    
}

    template<class T, std::size_t chunk_size>
    static std::complex<T> mul(std::complex<T> *arr, std::size_t count) {
        return implementation::mul<T, chunk_size>::compute(arr, count);
    }

}


