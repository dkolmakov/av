#pragma once

#include <complex>

#include "common.hpp"

namespace av_chunked {
    
namespace implementation {

    template <class T, std::size_t chunk_size>
    struct chunk_sum;
    
    template <class T>
    struct chunk_sum<T, 1> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr, const std::size_t count) {
            for (std::size_t i = 0; i < count; i++) {
                acc[0] += arr[i];
            }
        }
    };

    template <class T>
    struct chunk_sum<T, 2> {
        static force_inline void compute(std::complex<T> * acc, std::complex<T> * arr, const std::size_t count) {
            for (std::size_t i = 0; i < count; i += 2) {
                acc[0] += arr[i];
                acc[1] += arr[i + 1];
            }
        }
    };

    template <class T>
    struct chunk_sum<T, 4> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr, const std::size_t count) {
            for (std::size_t i = 0; i < count; i += 4) {
                acc[0] += arr[i];
                acc[1] += arr[i + 1];
                acc[2] += arr[i + 2];
                acc[3] += arr[i + 3];
            }
        }
    };

    template <class T>
    struct chunk_sum<T, 8> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr, const std::size_t count) {
            for (std::size_t i = 0; i < count; i += 8) {
                acc[0] += arr[i];
                acc[1] += arr[i + 1];
                acc[2] += arr[i + 2];
                acc[3] += arr[i + 3];
                acc[4] += arr[i + 4];
                acc[5] += arr[i + 5];
                acc[6] += arr[i + 6];
                acc[7] += arr[i + 7];
            }
        }
    };

    template <class T, std::size_t chunk_size>
    struct chunk_sum {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr, const std::size_t count) {
            for (std::size_t i = 0; i < count; i += 8) {
                acc[0] += arr[i];
                acc[1] += arr[i + 1];
                acc[2] += arr[i + 2];
                acc[3] += arr[i + 3];
                acc[4] += arr[i + 4];
                acc[5] += arr[i + 5];
                acc[6] += arr[i + 6];
                acc[7] += arr[i + 7];
            }
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
        implementation::chunk_sum<T, chunk_size>::compute(acc, arr, to_sum);
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


