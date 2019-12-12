#pragma once

#include <complex>

#include "common.hpp"

namespace av_chunked {
    
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

    template<class T, std::size_t chunk_size>
    static std::complex<T> sum(std::complex<T> *arr, std::size_t count) {
        // Specialized implementation
        std::complex<T> acc[chunk_size];
        const std::size_t to_sum = count - count % chunk_size;
        const std::size_t rem_offset = to_sum;
        
        for (std::size_t i = 0; i < chunk_size; i++)
            acc[i] = 0;
        
        // Sum by chunks
        asm volatile ("nop;nop;nop;");
        for (std::size_t i = 0; i < to_sum; i += chunk_size) {
            implementation::chunk_sum<T, chunk_size - 1>::compute(acc, arr + i);
        }
        asm volatile ("nop;nop;nop;");
        
        // Add the remainder
        std::complex<T> result(0,0);
        std::size_t j = 0;
        for (std::size_t i = rem_offset; i < count; i++, j++) {
            result += arr[i] + acc[j];
        }
        for (; j < chunk_size; j++) {
            result += acc[j];
        }
        
        return result;
    }

}


