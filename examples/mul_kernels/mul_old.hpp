#pragma once

#include <immintrin.h>
#include <complex>

#include "common.hpp"
#include "mul_old_avx.hpp"
#include "mul_old_sse.hpp"

namespace mul_old {
    
namespace impl {

#ifdef __AVX512F__
    constexpr std::size_t VALS_PER_OP = 4;
#elif __AVX__
    constexpr std::size_t VALS_PER_OP = 2;
#elif __SSE4_1__
    constexpr std::size_t VALS_PER_OP = 1;
#else
    constexpr std::size_t VALS_PER_OP = 1;
#endif
    
    template <class T, std::size_t chunk_size, std::size_t step, std::size_t parity_checker = (chunk_size * step) % VALS_PER_OP, std::size_t reg_size = av::SIMD_REG_SIZE>
    struct chunk_mul;
    
    template <class T, std::size_t chunk_size, std::size_t step>
    struct chunk_mul<T, chunk_size, step, 0, 32> {
        static force_inline void compute(std::complex<T> **acc, std::complex<T> **arr) {
            __m256d res[(chunk_size * step) / 2];
            avx::unpack<T, (chunk_size * step) / 2 - 1, step>::doIt(res, acc);

            __m256d v0[(chunk_size * step) / 2];
            avx::unpack<T, (chunk_size * step) / 2 - 1, step>::doIt(v0, arr);
            avx::multiply<T, (chunk_size * step) / 2 - 1>::doIt(res, v0);
            
            avx::pack<T, (chunk_size * step) / 2 - 1, step>::doIt(acc, res);
        }
    };
    
    template <class T, std::size_t chunk_size, std::size_t step>
    struct chunk_mul<T, chunk_size, step, 0, 16> {
        static force_inline void compute(std::complex<T> **acc, std::complex<T> **arr) {
            __m128d res[chunk_size];
            sse::unpack<T, chunk_size - 1, step>::doIt(res, acc);
            
            __m128d v0[chunk_size];
            sse::unpack<T, chunk_size - 1, step>::doIt(v0, arr);
            sse::multiply<T, chunk_size - 1>::doIt(res, v0);
            
            sse::pack<T, chunk_size - 1, step>::doIt(acc, res);
        }
    };
    
    template <class T, std::size_t chunk_size, std::size_t step, std::size_t parity_checker, std::size_t reg_size>
    struct chunk_mul {
        static force_inline void compute(std::complex<T> **acc, std::complex<T> **arr) {
            for (std::size_t i = 0; i < step; i++) {
                for (std::size_t j = 0; j < chunk_size; j++)
                    acc[i][j] *= arr[i][j];
            }
        }
    };
    
    
    template <class T, std::size_t chunk_size, std::size_t step, std::size_t index>
    struct unroll_chunks;

    template <class T, std::size_t chunk_size, std::size_t step>
    struct unroll_chunks<T, chunk_size, step, 0> {
        static force_inline void compute(std::complex<T> **left, std::complex<T> **right) {
            chunk_mul<T, chunk_size, step>::compute(&left[0], &right[0]);
        }
    };
    
    template <class T, std::size_t chunk_size, std::size_t step, std::size_t index>
    struct unroll_chunks {
        static force_inline void compute(std::complex<T> **left, std::complex<T> **right) {
            unroll_chunks<T, chunk_size, step, index - step>::compute(left, right);
            chunk_mul<T, chunk_size, step>::compute(&left[index], &right[index]);
        }
    };
    
}

    struct chunk_mul {
        static std::string get_label() {
            return "mul_old";
        }
        
        template <class T, std::size_t chunk_size, std::size_t n_chunks>
        struct core {
            constexpr static std::size_t step = (chunk_size < impl::VALS_PER_OP && (impl::VALS_PER_OP % chunk_size == 0) && (n_chunks % (impl::VALS_PER_OP / chunk_size)  == 0)) ? impl::VALS_PER_OP / chunk_size : 1;
            
            static force_inline void compute(std::complex<T> **left, std::complex<T> **right) {
                impl::unroll_chunks<T, chunk_size, step, n_chunks - step>::compute(left, right);
            }
        };
    };
}


