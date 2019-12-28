#pragma once

#include <complex>

#include "common.hpp"

namespace mul_unroll {
    
namespace implementation {

    template <class T, std::size_t index>
    struct chunk_mul;

    template <class T>
    struct chunk_mul<T, 0> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr) {
            acc[0] *= arr[0];
        }
    };
    
    template <class T, std::size_t index>
    struct chunk_mul {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr) {
            chunk_mul<T, index - 1>::compute(acc, arr);
            acc[index] *= arr[index];
        }
    };
}

    struct chunk_mul {
        static std::string get_label() {
            return "mul_unroll";
        }
        
        template <class T, std::size_t chunk_size>
        struct core {
            static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr) {
                implementation::chunk_mul<T, chunk_size>::compute(acc, arr);
            }
        };
    };
}


