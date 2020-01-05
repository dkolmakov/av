#pragma once

#include <complex>

#include "common.hpp"

namespace sum_unroll {
    
namespace implementation {

    template <class T, std::size_t index>
    struct chunk_sum;

    template <class T>
    struct chunk_sum<T, 0> {
        static force_inline void compute(T *acc, T *arr) {
            acc[0] += arr[0];
        }
    };

    template <class T, std::size_t index>
    struct chunk_sum {
        static force_inline void compute(T *acc, T *arr) {
            chunk_sum<T, index - 1>::compute(acc, arr);
            acc[index] += arr[index];
        }
    };
    
    template <class T, std::size_t chunk_size, std::size_t index>
    struct unroll_chunks;

    template <class T, std::size_t chunk_size>
    struct unroll_chunks<T, chunk_size, 0> {
        static force_inline void compute(std::complex<T> **left, std::complex<T> **right) {
            chunk_sum<T, 2 * chunk_size - 1>::compute((T *)left[0], (T *)right[0]);
        }
    };
    
    template <class T, std::size_t chunk_size, std::size_t index>
    struct unroll_chunks {
        static force_inline void compute(std::complex<T> **left, std::complex<T> **right) {
            unroll_chunks<T, chunk_size, index - 1>::compute(left, right);
            chunk_sum<T, 2 * chunk_size - 1>::compute((T *)left[index], (T *)right[index]);
        }
    };    
}

    struct chunk_sum {
        static std::string get_label() {
            return "sum_unroll";
        }
        
        template <class T, std::size_t chunk_size, std::size_t n_chunks>
        struct core {
            static force_inline void compute(std::complex<T> **left, std::complex<T> **right) {
                implementation::unroll_chunks<T, chunk_size, n_chunks - 1>::compute(left, right);
            }
        };
    };
}


