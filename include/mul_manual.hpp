#pragma once

#include <immintrin.h>
#include <complex>

#include "common.hpp"

namespace av_mul_manual {
    
namespace implementation {

    template <class T, std::size_t chunk_size>
    struct chunk_mul;

    template <class T>
    struct chunk_mul<T, 0> {
        static force_inline void compute(std::complex<T> *result, std::complex<T> *arr, std::size_t count) {
            // Default implementation
            std::complex<T> res(1,0);
            for (std::size_t i = 0; i < count; i++) {
                res *= arr[i];
            }
            *result = res;
        }
    };
    
    template <class T>
    struct chunk_mul<T, 1> {
        static force_inline void compute(std::complex<T> *result, std::complex<T> *arr, std::size_t count) {
            __m128d res = _mm_load_pd(reinterpret_cast<double *>(arr));
            
            for (std::size_t i = 1; i < count; i += 1) {
                __m128d v0 = _mm_load_pd1(reinterpret_cast<double *>(arr + i));
                __m128d v1 = _mm_load_pd1(reinterpret_cast<double *>(arr + i) + 1);
                res = _mm_addsub_pd(_mm_mul_pd(v0, res), _mm_shuffle_pd(_mm_mul_pd(v1, res), _mm_mul_pd(v1, res), 1));
            }
            
            _mm_store_pd(reinterpret_cast<double *>(result), res);
        }
    };

    template <class T>
    struct chunk_mul<T, 2> {
        static force_inline void compute(std::complex<T> *result, std::complex<T> *arr, std::size_t count) {
            std::cout << "No solution for AVX yet" << std::endl;
            chunk_mul<T, 0>::compute(result, arr, count);
            result[1] = {1,0};
        }
    };

    template <class T>
    struct chunk_mul<T, 4> {
        static force_inline void compute(std::complex<T> *result, std::complex<T> *arr, std::size_t count) {
            std::cout << "No solution for AVX512 yet" << std::endl;
            chunk_mul<T, 0>::compute(result, arr, count);
            result[1] = {1,0};
            result[2] = {1,0};
            result[3] = {1,0};
        }
    };

    template <class T, std::size_t chunk_size = av::SIMD_REG_SIZE / sizeof(T) / 2>
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
            std::complex<T> acc[chunk_size];
            std::size_t to_mul = count - count % chunk_size;
            
            // Sum by chunks
            asm volatile ("nop;nop;nop;");
            chunk_mul<T, chunk_size>::compute(acc, arr, to_mul);
            asm volatile ("nop;nop;nop;");

            std::size_t i = to_mul;
            
            // Add the remainder
            std::complex<T> result(1,0);
            std::size_t j = 0;
            for (; i < count; i++, j++) {
                result *= arr[i] * acc[j];
            }
            for (; j < chunk_size; j++) {
                result *= acc[j];
            }
            
            return result;
        }
    };
    
}

    template<class T>
    static std::complex<T> mul(std::complex<T> *arr, std::size_t count) {
        return implementation::mul<T>::compute(arr, count);
    }

}


