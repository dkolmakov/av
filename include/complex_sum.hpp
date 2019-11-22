#pragma once

#include <complex>

namespace av {
    
namespace implementation {

    #define force_inline inline __attribute__((always_inline))
    
#ifdef __AVX512F__
    constexpr std::size_t SIMD_REG_SIZE = 64;
#elif __AVX__
    constexpr std::size_t SIMD_REG_SIZE = 32;
#elif __SSE4_1__
    constexpr std::size_t SIMD_REG_SIZE = 16;
#else
    constexpr std::size_t SIMD_REG_SIZE = 0;
#endif
    
    template <class T, std::size_t chunk_size>
    struct chunk_sum;
    
    template <class T>
    struct chunk_sum<T, 2> {
        static forse_inline std::complex<T> compute(std::complex<T> *arr) {
            T real = arr[0].real() + arr[1].real();
            T imag = arr[0].imag() + arr[1].imag();
            
            return std::complex<T>(real, imag);
        }
    };

    template <class T, std::size_t chunk_size = SIMD_REG_SIZE / sizeof(T)>
    struct sum;
    
    template <class T>
    struct sum<T, 0> {
        static force_inline std::complex<T> compute(std::complex<T> *arr, std::size_t count) {
            // Default implementation
            
        }
    };
    
    template <class T, std::size_t chunk_size>
    struct sum {
        static force_inline std::complex<T> compute(std::complex<T> *arr, std::size_t count) {
            // Specialized implementation
            std::complex<T> result(0,0);
            
            result += chunk_sum<T, chunk_size>::compute(arr);
            
        }
    };
    
}
    template<class T>
    std::complex<T> sum(std::complex<T> *arr, std::size_t count) {
        return implementation::sum<T>::compute(arr, count);
    }
}


