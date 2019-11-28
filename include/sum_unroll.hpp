#pragma once

#include <complex>

#include "common.hpp"

namespace av_unroll {
    
namespace implementation {

    template <class T, std::size_t chunk_size>
    struct chunk_sum;
    
    template <class T>
    struct chunk_sum<T, 2> {
        static force_inline std::complex<T> compute(std::complex<T> *arr) {
            T real = arr[0].real() + arr[1].real();
            T imag = arr[0].imag() + arr[1].imag();
            
            return std::complex<T>(real, imag);
        }
    };

    template <class T>
    struct chunk_sum<T, 4> {
        static force_inline std::complex<T> compute(std::complex<T> *arr) {
            T real_0 = arr[0].real() + arr[1].real();
            T real_1 = arr[2].real() + arr[3].real();
            T imag_0 = arr[0].imag() + arr[1].imag();
            T imag_1 = arr[2].imag() + arr[3].imag();
            T real = real_0 + real_1;
            T imag = imag_0 + imag_1;

            return std::complex<T>(real, imag);
        }
    };

    template <class T>
    struct chunk_sum<T, 8> {
        static force_inline std::complex<T> compute(std::complex<T> *arr) {
            T real_0 = arr[0].real() + arr[1].real();
            T real_1 = arr[2].real() + arr[3].real();
            T real_2 = arr[4].real() + arr[5].real();
            T real_3 = arr[6].real() + arr[7].real();
            T imag_0 = arr[0].imag() + arr[1].imag();
            T imag_1 = arr[2].imag() + arr[3].imag();
            T imag_2 = arr[4].imag() + arr[5].imag();
            T imag_3 = arr[6].imag() + arr[7].imag();
            T real = real_0 + real_1 + real_2 + real_3;
            T imag = imag_0 + imag_1 + imag_2 + imag_3;
        
            return std::complex<T>(real, imag);
        }
    };

    template <class T>
    struct chunk_sum<T, 16> {
        static force_inline std::complex<T> compute(std::complex<T> *arr) {
            T real_0 = arr[0].real() + arr[1].real();
            T imag_0 = arr[0].imag() + arr[1].imag();
            T real_1 = arr[2].real() + arr[3].real();
            T imag_1 = arr[2].imag() + arr[3].imag();
            T real_2 = arr[4].real() + arr[5].real();
            T imag_2 = arr[4].imag() + arr[5].imag();
            T real_3 = arr[6].real() + arr[7].real();
            T imag_3 = arr[6].imag() + arr[7].imag();
            T real_4 = arr[8].real() + arr[9].real();
            T imag_4 = arr[8].imag() + arr[9].imag();
            T real_5 = arr[10].real() + arr[11].real();
            T imag_5 = arr[10].imag() + arr[11].imag();
            T real_6 = arr[12].real() + arr[13].real();
            T imag_6 = arr[12].imag() + arr[13].imag();
            T real_7 = arr[14].real() + arr[15].real();
            T imag_7 = arr[14].imag() + arr[15].imag();
            T real = real_0 + real_1 + real_2 + real_3 + real_4 + real_5 + real_6 + real_7;
            T imag = imag_0 + imag_1 + imag_2 + imag_3 + imag_4 + imag_5 + imag_6 + imag_7;
            
            return std::complex<T>(real, imag);
        }
    };
    
    template <class T, std::size_t chunk_size = av::SIMD_REG_SIZE / sizeof(T) >
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
            std::complex<T> result(0,0);
            
            asm volatile ("nop;nop;nop;");
            // Sum by chunks
            std::size_t i = 0;
            while (i + chunk_size < count) {
                result += chunk_sum<T, chunk_size>::compute(arr + i);
                i += chunk_size;
            }
            asm volatile ("nop;nop;nop;");
            
            // Add the remainder
            for (; i < count; i++) {
                result += arr[i];
            }
            
            return result;
        }
    };
    
}

    template<class T>
    static std::complex<T> sum(std::complex<T> *arr, std::size_t count) {
        return implementation::sum<T>::compute(arr, count);
    }

    template<class T>
    static std::complex<T> sum_default(std::complex<T> *arr, std::size_t count) {
        return implementation::sum<T, 0>::compute(arr, count);
    }
    
}


