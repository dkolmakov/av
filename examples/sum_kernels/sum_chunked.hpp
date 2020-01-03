#pragma once

#include <complex>

#include "common.hpp"

namespace sum_chunked {
    
namespace implementation {

    template <class T, std::size_t index>
    struct chunk_sum;

    template <class T>
    struct chunk_sum<T, 0> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr) {
            acc[0] += arr[0];
        }
    };
    
    template <class T, std::size_t index>
    struct chunk_sum {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr) {
            chunk_sum<T, index - 1>::compute(acc, arr);
            acc[index] += arr[index];
        }
    };

}

    struct chunk_sum {
        static std::string get_label() {
            return "sum_chunked";
        }
        
        template <class T, std::size_t chunk_size>
        struct core {
            static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr) {
                implementation::chunk_sum<T, chunk_size - 1>::compute(acc, arr);
            }
        };
    };
}


