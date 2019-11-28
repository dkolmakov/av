#pragma once

#include <complex>

#include "common.hpp"

namespace av_chunked {
    
namespace implementation {

    template <class T, std::size_t chunk_size>
    struct chunk_sum;
    
    template <class T>
    struct chunk_sum<T, 1> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr, std::size_t count) {
            for (std::size_t i = 0; i < count; i++) {
                acc[0] += arr[i];
            }
        }
    };

    template <class T>
    struct chunk_sum<T, 2> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr, std::size_t count) {
            for (std::size_t i = 0; i < count; i += 2) {
                acc[0] += arr[i];
                acc[1] += arr[i + 1];
            }
        }
    };

    template <class T>
    struct chunk_sum<T, 4> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr, std::size_t count) {
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
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr, std::size_t count) {
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
    
    template <class T, std::size_t chunk_size = av::SIMD_REG_SIZE / sizeof(T) / 2>
    struct sum;
    
    template <class T>
    struct sum<T, 0> {
        static force_inline std::complex<T> compute(std::complex<T> *arr, const std::size_t count) {
            // Default implementation
            std::complex<T> result(0,0);
            
            asm volatile ("nop;nop;nop;");
            for (std::size_t i = 0; i < count; i++) {
                result += arr[i];
            }
            asm volatile ("nop;nop;nop;");
            
            return result;
        }
    };
    
    template <class T, std::size_t chunk_size>
    struct sum {
        static force_inline std::complex<T> compute(std::complex<T> *arr, std::size_t count) {
            // Specialized implementation
            std::complex<T> acc[chunk_size];
            std::size_t i = 0;
            for (; i < chunk_size; i++)
                acc[i] = arr[i];
            std::size_t to_sum = count - chunk_size - count % chunk_size;
            
            asm volatile ("nop;nop;nop;");
            // Sum by chunks
            chunk_sum<T, chunk_size>::compute(acc, arr + i, to_sum);
            asm volatile ("nop;nop;nop;");
            i += to_sum;
            
            // Add the remainder
            std::complex<T> result(0,0);
            std::size_t j = 0;
            for (; i < count; i++, j++) {
                result += arr[i] + acc[j];
            }
            for (; j < chunk_size; j++) {
                result += acc[j];
            }
            
            return result;
        }
    };
    
}

    template<class T>
    static std::complex<T> sum(std::complex<T> *arr, std::size_t count) {
        return implementation::sum<T>::compute(arr, count);
    }

}


