#pragma once

#include <immintrin.h>
#include <complex>

#include "common.hpp"
#include "mul_old_avx.hpp"
#include "mul_old_sse.hpp"

namespace mul_old {
    
namespace implementation {

#ifdef __AVX512F__
    constexpr std::size_t NUMBERS_PER_OP = 4;
#elif __AVX__
    constexpr std::size_t NUMBERS_PER_OP = 2;
#elif __SSE4_1__
    constexpr std::size_t NUMBERS_PER_OP = 1;
#else
    constexpr std::size_t NUMBERS_PER_OP = 1;
#endif
    
    template <class T, std::size_t chunk_size, std::size_t parity_checker = chunk_size % NUMBERS_PER_OP, std::size_t reg_size = av::SIMD_REG_SIZE>
    struct chunk_mul;
    
    template <class T, std::size_t chunk_size>
    struct chunk_mul<T, chunk_size, 0, 32> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr) {
            __m256d res[chunk_size / 2];
            avx::unpack<T, chunk_size / 2 - 1>::doIt(res, acc);
            
            __m256d v0[chunk_size / 2];
            avx::unpack<T, chunk_size / 2 - 1>::doIt(v0, arr);
            avx::multiply<T, chunk_size / 2 - 1>::doIt(res, v0);
            
            avx::pack<T, chunk_size / 2 - 1>::doIt(acc, res);
        }
    };
    
    template <class T, std::size_t chunk_size>
    struct chunk_mul<T, chunk_size, 0, 16> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr) {
            __m128d res[chunk_size];
            sse::unpack<T, chunk_size - 1>::doIt(res, acc);
            
            __m128d v0[chunk_size];
            sse::unpack<T, chunk_size - 1>::doIt(v0, arr);
            sse::multiply<T, chunk_size - 1>::doIt(res, v0);
            
            sse::pack<T, chunk_size - 1>::doIt(acc, res);
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
            return "mul_old";
        }
        
        template <class T, std::size_t chunk_size, std::size_t n_chunks>
        struct core {
            static force_inline void compute(std::complex<T> **left, std::complex<T> **right) {
                implementation::unroll_chunks<T, chunk_size, n_chunks - 1>::compute(left, right);
            }
        };
    };
}


