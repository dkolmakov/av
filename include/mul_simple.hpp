#pragma once

#include <complex>

#include "common.hpp"

namespace av_mul_simple {
    
namespace implementation {

    template <class T>
    struct mul {
        static force_inline std::complex<T> compute(std::complex<T> *arr, const std::size_t count) {
            // Default implementation
            std::complex<T> result(1,0);
            
            asm volatile ("nop;nop;nop;");
            for (std::size_t i = 0; i < count; i++) {
                result *= arr[i];
            }
            asm volatile ("nop;nop;nop;");
            
            return result;
        }
    };
    
}

    template<class T>
    static std::complex<T> mul(std::complex<T> *arr, std::size_t count) {
        return implementation::mul<T>::compute(arr, count);
    }

}


