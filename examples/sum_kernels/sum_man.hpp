#pragma once

#include <immintrin.h>
#include <complex>

#include "common.hpp"
#include "sum_simple.hpp"
#include "sum_man_avx.hpp"
#include "sum_man_sse.hpp"

namespace sum_man {

namespace impl {

    template <class T, 
              std::size_t chunk_size, 
              std::size_t step, 
              std::size_t vals_per_op, 
              std::size_t parity_checker = (chunk_size * step) % vals_per_op, 
              std::size_t reg_size = av::SIMD_REG_SIZE>
    struct chunk_sum;
    
    template <std::size_t chunk_size, std::size_t step, std::size_t vals_per_op>
    struct chunk_sum<std::complex<double>, chunk_size, step, vals_per_op, 0, 32> {
        constexpr static std::size_t portion_size = chunk_size * step;
        
        static force_inline void compute(std::complex<double> **acc, std::complex<double> **left, std::complex<double> **right) {
            __m256d dataA[portion_size / vals_per_op];
            avx::unpack<std::complex<double>, portion_size / vals_per_op - 1, step>::doIt(dataA, left);
            __m256d dataB[portion_size / vals_per_op];
            avx::unpack<std::complex<double>, portion_size / vals_per_op - 1, step>::doIt(dataB, right);
            
            avx::summation<std::complex<double>, portion_size / vals_per_op - 1>::doIt(dataA, dataB);

            avx::pack<std::complex<double>, portion_size / vals_per_op - 1, step>::doIt(acc, dataA);
        }
    };

    template <std::size_t chunk_size, std::size_t step, std::size_t vals_per_op>
    struct chunk_sum<std::complex<float>, chunk_size, step, vals_per_op, 0, 32> {
        constexpr static std::size_t portion_size = chunk_size * step;
        
        static force_inline void compute(std::complex<float> **acc, std::complex<float> **left, std::complex<float> **right) {
            __m256 dataA[portion_size / vals_per_op];
            avx::unpack<std::complex<float>, portion_size / vals_per_op - 1, step>::doIt(dataA, left);
            __m256 dataB[portion_size / vals_per_op];
            avx::unpack<std::complex<float>, portion_size / vals_per_op - 1, step>::doIt(dataB, right);
            
            avx::summation<std::complex<float>, portion_size / vals_per_op - 1>::doIt(dataA, dataB);

            avx::pack<std::complex<float>, portion_size / vals_per_op - 1, step>::doIt(acc, dataA);
        }
    };

    template <class T, std::size_t chunk_size, std::size_t step, std::size_t vals_per_op>
    struct chunk_sum<T, chunk_size, step, vals_per_op, 0, 16> {
        constexpr static std::size_t portion_size = chunk_size * step;
        
        static force_inline void compute(T **acc, T **left, T **right) {
            __m128d dataA[portion_size / vals_per_op];
            sse::unpack<T, portion_size / vals_per_op - 1, step>::doIt(dataA, left);
            __m128d dataB[portion_size / vals_per_op];
            sse::unpack<T, portion_size / vals_per_op - 1, step>::doIt(dataB, right);
            
            sse::summation<T, portion_size / vals_per_op - 1>::doIt(dataA, dataB);

            sse::pack<T, portion_size / vals_per_op - 1, step>::doIt(acc, dataA);
        }
    };
    
    template <class T, std::size_t chunk_size, std::size_t step, std::size_t, std::size_t, std::size_t>
    struct chunk_sum {
        static force_inline void compute(T **acc, T **left, T **right) {
            sum_simple::chunk_sum::core<T, chunk_size, step>::compute(acc, left, right);
        }
    };

    template <class T, std::size_t chunk_size, std::size_t step, std::size_t vals_per_op, std::size_t index>
    struct unroll_chunks;

    template <class T, std::size_t chunk_size, std::size_t step, std::size_t vals_per_op>
    struct unroll_chunks<T, chunk_size, step, vals_per_op, 0> {
        static force_inline void compute(T **acc, T **left, T **right) {
            chunk_sum<T, chunk_size, step, vals_per_op>::compute(&acc[0], &left[0], &right[0]);
        }
    };
    
    template <class T, std::size_t chunk_size, std::size_t step, std::size_t vals_per_op, std::size_t index>
    struct unroll_chunks {
        static force_inline void compute(T **acc, T **left, T **right) {
            unroll_chunks<T, chunk_size, step, vals_per_op, index - step>::compute(acc, left, right);
            chunk_sum<T, chunk_size, step, vals_per_op>::compute(&acc[index], &left[index], &right[index]);
        }
    };
}

    struct chunk_sum {
        static std::string get_label() {
            return "sum_man";
        }
        
        template <class T, std::size_t chunk_size, std::size_t n_chunks>
        struct core {
            constexpr static std::size_t vals_per_op = (av::SIMD_REG_SIZE / sizeof(T) == 0) ? 1 : av::SIMD_REG_SIZE / sizeof(T);
            constexpr static std::size_t step = (chunk_size < vals_per_op && (vals_per_op % chunk_size == 0) && (n_chunks % (vals_per_op / chunk_size)  == 0)) ? vals_per_op / chunk_size : 1;
            
            static force_inline void compute(T **acc, T **left, T **right) {
                impl::unroll_chunks<T, chunk_size, step, vals_per_op, n_chunks - step>::compute(acc, left, right);
            }
        };
    };
   
}


