#pragma once

#include <complex>

#include "common.hpp"

namespace av_mul_unroll {
    
namespace implementation {

    template <class T, std::size_t chunk_size>
    struct chunk_mul;
    
    template <class T>
    struct chunk_mul<T, 2> {
        static force_inline std::complex<T> compute(std::complex<T> *arr) {
            T rr = arr[0].real() * arr[1].real();
            T ii = arr[0].imag() * arr[1].imag();
            T ri = arr[0].real() * arr[1].imag();
            T ir = arr[0].imag() * arr[1].real();
            
            return std::complex<T>(rr - ii, ri + ir);
        }
    };

    template <class T>
    struct chunk_mul<T, 4> {
        static force_inline std::complex<T> compute(std::complex<T> *arr) {
            T rr_0 = arr[0].real() * arr[1].real();
            T ii_0 = arr[0].imag() * arr[1].imag();
            T ri_0 = arr[0].real() * arr[1].imag();
            T ir_0 = arr[0].imag() * arr[1].real();

            T rr_1 = arr[2].real() * arr[3].real();
            T ii_1 = arr[2].imag() * arr[3].imag();
            T ri_1 = arr[2].real() * arr[3].imag();
            T ir_1 = arr[2].imag() * arr[3].real();
            
            T real_0 = rr_0 - ii_0;
            T real_1 = rr_1 - ii_1;
            T imag_0 = ri_0 + ir_0;
            T imag_1 = ri_1 + ir_1;
            
            T real = real_0 * real_1 - imag_0 * imag_1;
            T imag = real_0 * imag_1 + real_1 * imag_0;

            return std::complex<T>(real, imag);
        }
    };

    template <class T>
    struct chunk_mul<T, 8> {
        static force_inline std::complex<T> compute(std::complex<T> *arr) {
            std::complex<T> result(1,0);
            
            for (std::size_t i = 0; i < 8; i++) {
                result *= arr[i];
            }
        
            return result;
        }
    };

    template <class T, std::size_t chunk_size = av::SIMD_REG_SIZE / sizeof(T) >
    struct mul;
    
    template <class T>
    struct mul<T, 0> {
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
    
    template <class T, std::size_t chunk_size>
    struct mul {
        static force_inline std::complex<T> compute(std::complex<T> *arr, std::size_t count) {
            // Specialized implementation
            std::complex<T> result(1,0);
            
            asm volatile ("nop;nop;nop;");
            // Sum by chunks
            std::size_t i = 0;
            while (i + chunk_size < count) {
                result *= chunk_mul<T, chunk_size>::compute(arr + i);
                i += chunk_size;
            }
            asm volatile ("nop;nop;nop;");
            
            // Add the remainder
            for (; i < count; i++) {
                result *= arr[i];
            }
            
            return result;
        }
    };
    
}

    template<class T>
    static std::complex<T> mul(std::complex<T> *arr, std::size_t count) {
        return implementation::mul<T>::compute(arr, count);
    }

    template<class T, std::size_t chunk_size>
    struct ToTest {
        static std::complex<T> to_test(std::complex<T> *arr, std::size_t count) {
            return mul<T>(arr, count);
        }
    };

}


