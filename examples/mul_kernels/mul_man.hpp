#pragma once

#include <immintrin.h>
#include <complex>

#include "common.hpp"
#include "mul_avx.hpp"
#include "mul_sse.hpp"

namespace mul_man {
    
namespace implementation {

#ifdef __AVX512F__
    constexpr std::size_t NUMBERS_PER_OP = 8;
#elif __AVX__
    constexpr std::size_t NUMBERS_PER_OP = 4;
#elif __SSE4_1__
    constexpr std::size_t NUMBERS_PER_OP = 2;
#else
    constexpr std::size_t NUMBERS_PER_OP = 1;
#endif
    
    template <class T, std::size_t chunk_size, std::size_t parity_checker = chunk_size % NUMBERS_PER_OP, std::size_t reg_size = av::SIMD_REG_SIZE>
    struct chunk_mul;
    
    template <class T, std::size_t chunk_size>
    struct chunk_mul<T, chunk_size, 0, 32> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr) {
            __m256d realA[chunk_size / 4];
            __m256d imagA[chunk_size / 4]; 
            avx::unpack_complex<T, chunk_size / 4 - 1>::unpack(realA, imagA, acc);
            
            __m256d realB[chunk_size / 4];
            __m256d imagB[chunk_size / 4];
            avx::unpack_complex<T, chunk_size / 4 - 1>::unpack(realB, imagB, arr);

            avx::multiply<T, chunk_size / 4 - 1>::compute(realA, imagA, realB, imagB);
            avx::pack<T, chunk_size / 4 - 1>::make(realA, imagA, acc);
        }
    };

    template <class T, std::size_t chunk_size>
    struct chunk_mul<T, chunk_size, 0, 16> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr) {
            __m128d realA[chunk_size / 2];
            __m128d imagA[chunk_size / 2]; 
            sse::unpack_complex<T, chunk_size / 2 - 1>::unpack(realA, imagA, acc);
            
            __m128d realB[chunk_size / 2];
            __m128d imagB[chunk_size / 2];
            sse::unpack_complex<T, chunk_size / 2 - 1>::unpack(realB, imagB, arr);

            sse::multiply<T, chunk_size / 2 - 1>::compute(realA, imagA, realB, imagB);
            sse::pack<T, chunk_size / 2 - 1>::make(realA, imagA, acc);
        }
    };
    
    template <class T, std::size_t chunk_size, std::size_t parity_checker, std::size_t reg_size>
    struct chunk_mul {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr) {
            for (std::size_t i = 0; i < chunk_size; i++)
                acc[i] *= arr[i];
        }
    };
    
    
    template <class T, std::size_t chunk_size, std::size_t index>
    struct unroll_chunks;

    template <class T, std::size_t chunk_size>
    struct unroll_chunks<T, chunk_size, 0> {
        static force_inline void compute(std::complex<T> **left, std::complex<T> **right) {
            chunk_mul<T, chunk_size>::compute(left[0], right[0]);
        }
    };
    
    template <class T, std::size_t chunk_size, std::size_t index>
    struct unroll_chunks {
        static force_inline void compute(std::complex<T> **left, std::complex<T> **right) {
            unroll_chunks<T, chunk_size, index - 1>::compute(left, right);
            chunk_mul<T, chunk_size>::compute(left[index], right[index]);
        }
    };
    
}

    struct chunk_mul {
        static std::string get_label() {
            return "mul_man";
        }
        
        template <class T, std::size_t chunk_size, std::size_t n_chunks>
        struct core {
            static force_inline void compute(std::complex<T> **left, std::complex<T> **right) {
                implementation::unroll_chunks<T, chunk_size, n_chunks - 1>::compute(left, right);
            }
        };
    };
}


