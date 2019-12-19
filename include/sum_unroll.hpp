#pragma once

#include <complex>

#include "common.hpp"

namespace sum_unroll {
    
namespace implementation {

    template <class T, std::size_t chunk_size>
    struct real_sum;
    
    template <class T>
    struct real_sum<T, 2> {
        static force_inline T compute(std::complex<T> *arr) {
            return arr[0].real() + arr[1].real();
        }        
    };
    
    template <class T, std::size_t chunk_size>
    struct real_sum {
        static force_inline T compute(std::complex<T> *arr) {
            T real0 = real_sum<T, chunk_size - 2>::compute(arr);
            T real1 = arr[chunk_size - 2].real() + arr[chunk_size - 1].real();
            return real0 + real1;
        }
    };

    template <class T, std::size_t chunk_size>
    struct imag_sum;
    
    template <class T>
    struct imag_sum<T, 2> {
        static force_inline T compute(std::complex<T> *arr) {
            return arr[0].imag() + arr[1].imag();
        }        
    };
    
    template <class T, std::size_t chunk_size>
    struct imag_sum {
        static force_inline T compute(std::complex<T> *arr) {
            T real0 = imag_sum<T, chunk_size - 2>::compute(arr);
            T real1 = arr[chunk_size - 2].imag() + arr[chunk_size - 1].imag();
            return real0 + real1;
        }
    };

    template <class T, std::size_t chunk_size, std::size_t parity_checker>
    struct chunk_sum;
    
    template <class T, std::size_t chunk_size>
    struct chunk_sum<T, chunk_size, 0> {
        static force_inline std::complex<T> compute(std::complex<T> *arr) {
            T real = real_sum<T, chunk_size>::compute(arr);
            T imag = imag_sum<T, chunk_size>::compute(arr);
            
            return std::complex<T>(real, imag);
        }
    };

    template <class T, std::size_t chunk_size, std::size_t parity_checker>
    struct chunk_sum {
        static force_inline std::complex<T> compute(std::complex<T> *arr) {
            // Default implementation
            std::complex<T> result(0,0);
            
            for (std::size_t i = 0; i < chunk_size; i++) {
                result += arr[i];
            }

            return result;
        }
    };
}

    template<class T, std::size_t chunk_size>
    static std::complex<T> sum(std::complex<T> *arr, std::size_t count) {
        std::complex<T> result(0,0);
        std::size_t i = 0;
        
        // Sum by chunks
        asm volatile ("nop;nop;nop;");
        for (; i + chunk_size < count; i += chunk_size) {
            result += implementation::chunk_sum<T, chunk_size, chunk_size % 2>::compute(arr + i);
        }
        asm volatile ("nop;nop;nop;");
        
        // Add the remainder
        for (; i < count; i++) {
            result += arr[i];
        }
        
        return result;
    }

    template<class T, std::size_t chunk_size>
    struct ToTest {
        static std::complex<T> to_test(std::complex<T> *arr, std::size_t count) {
            return sum<T, chunk_size>(arr, count);
        }
    };
}


