#pragma once

#include <immintrin.h>
#include <complex>

#include "common.hpp"
#include "sum_man_avx.hpp"
#include "sum_man_sse.hpp"

namespace sum_man {

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
    struct chunk_sum;
    
    template <class T, std::size_t chunk_size, std::size_t step>
    struct chunk_sum<T, chunk_size, step, 0, 32> {
        constexpr static std::size_t portion_size = chunk_size * step;
        
        static force_inline void compute(std::complex<T> **acc, std::complex<T> **arr) {
            __m256d dataA[portion_size / VALS_PER_OP];
            avx::unpack<T, portion_size / VALS_PER_OP - 1, step>::doIt(dataA, acc);
            __m256d dataB[portion_size / VALS_PER_OP];
            avx::unpack<T, portion_size / VALS_PER_OP - 1, step>::doIt(dataB, arr);
            
            avx::summation<T, portion_size / VALS_PER_OP - 1>::doIt(dataA, dataB);

            avx::pack<T, portion_size / VALS_PER_OP - 1, step>::doIt(acc, dataA);
        }
    };

    template <class T, std::size_t chunk_size, std::size_t step>
    struct chunk_sum<T, chunk_size, step, 0, 16> {
        constexpr static std::size_t portion_size = chunk_size * step;
        
        static force_inline void compute(std::complex<T> **acc, std::complex<T> **arr) {
            __m128d dataA[portion_size / VALS_PER_OP];
            sse::unpack<T, portion_size / VALS_PER_OP - 1, step>::doIt(dataA, acc);
            __m128d dataB[portion_size / VALS_PER_OP];
            sse::unpack<T, portion_size / VALS_PER_OP - 1, step>::doIt(dataB, arr);
            
            sse::summation<T, portion_size / VALS_PER_OP - 1>::doIt(dataA, dataB);

            sse::pack<T, portion_size / VALS_PER_OP - 1, step>::doIt(acc, dataA);
        }
    };
    
    template <class T, std::size_t chunk_size, std::size_t step, std::size_t, std::size_t>
    struct chunk_sum {
        static force_inline void compute(std::complex<T> **acc, std::complex<T> **arr) {
            for (std::size_t i = 0; i < step; i++) {
                for (std::size_t j = 0; j < chunk_size; j++)
                    acc[i][j] += arr[i][j];
            }
        }
    };

    template <class T, std::size_t chunk_size, std::size_t step, std::size_t index>
    struct unroll_chunks;

    template <class T, std::size_t chunk_size, std::size_t step>
    struct unroll_chunks<T, chunk_size, step, 0> {
        static force_inline void compute(std::complex<T> **left, std::complex<T> **right) {
            chunk_sum<T, chunk_size, step>::compute(&left[0], &right[0]);
        }
    };
    
    template <class T, std::size_t chunk_size, std::size_t step, std::size_t index>
    struct unroll_chunks {
        static force_inline void compute(std::complex<T> **left, std::complex<T> **right) {
            unroll_chunks<T, chunk_size, step, index - step>::compute(left, right);
            chunk_sum<T, chunk_size, step>::compute(&left[index], &right[index]);
        }
    };
}

    struct chunk_sum {
        static std::string get_label() {
            return "sum_man";
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


