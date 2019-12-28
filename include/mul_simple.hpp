#pragma once

#include <complex>

#include "common.hpp"

namespace mul_simple {
    
    struct chunk_mul {
        static std::string get_label() {
            return "mul_simple";
        }
        
        template <class T, std::size_t chunk_size>
        struct core {
            static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr) {
                for (std::size_t i = 0; i < chunk_size; i++)
                    acc[i] *= arr[i];
            }
        };
    };
}


