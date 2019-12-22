#pragma once

#include <immintrin.h>
#include <complex>

#include "common.hpp"

namespace sum_man_avx {
    
namespace implementation {

    template <class T, std::size_t index>
    struct unpack;

    template <class T>
    struct unpack<T, 0> {
        static force_inline void doIt(__m256d *vals, std::complex<T> *arr) {
            vals[0] = _mm256_loadu_pd((double*)(arr + 0 * 2));
        }
    };

    template <class T, std::size_t index>
    struct unpack {
        static force_inline void doIt(__m256d *vals, std::complex<T> *arr) {
            vals[index] = _mm256_loadu_pd((double*)(arr + index * 2));
            unpack<T, index - 1>::doIt(vals, arr);
        }
    };

    template <class T, std::size_t index>
    struct summation;
    
    template <class T>
    struct summation<T, 0> {
        static force_inline void doIt(__m256d *acc, __m256d *vals) {
            acc[0] = _mm256_add_pd(acc[0], vals[0]);
        }
    };
    
    template <class T, std::size_t index>
    struct summation {
        static force_inline void doIt(__m256d *acc, __m256d *vals) {
            acc[index] = _mm256_add_pd(acc[index], vals[index]);

            summation<T, index - 1>::doIt(acc, vals);
        }
    };
    
    template <class T, std::size_t index>
    struct pack;

    template <class T>
    struct pack<T, 0> {
        static force_inline void doIt(std::complex<T> *dst, __m256d *acc) {
            _mm256_storeu_pd((double*)(dst + 0 * 2), acc[0]);
        }
    };

    template <class T, std::size_t index>
    struct pack {
        static force_inline void doIt(std::complex<T> *dst, __m256d *acc) {
            _mm256_storeu_pd((double*)(dst + index * 2), acc[index]);
            pack<T, index - 1>::doIt(dst, acc);
        }
    };

    
    template <class T, std::size_t chunk_size, std::size_t parity_checker>
    struct chunk_sum;
    
    template <class T, std::size_t chunk_size>
    struct chunk_sum<T, chunk_size, 0> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr, std::size_t count) {
            __m256d dataA[chunk_size];
            unpack<T, chunk_size - 1>::doIt(dataA, arr);
            
            for (std::size_t i = chunk_size; i < count; i += chunk_size) {
                __m256d dataB[chunk_size];
                
                unpack<T, chunk_size - 1>::doIt(dataB, arr + i);
                summation<T, chunk_size - 1>::doIt(dataA, dataB);
            }

            pack<T, chunk_size - 1>::doIt(acc, dataA);
        }
    };

    template <class T, std::size_t chunk_size, std::size_t parity_checker>
    struct chunk_sum {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr, std::size_t count) {
            std::complex<T> result(0,0);
            for (std::size_t i = 0; i < count; i++) {
                result += arr[i];
            }
            *acc = result;
        }
    };

    template <class T, std::size_t chunk_size, std::size_t reg_size = av::SIMD_REG_SIZE >
    struct sum;
    
    template <class T, std::size_t chunk_size>
    struct sum<T, chunk_size, 32> {
        static force_inline std::complex<T> compute(std::complex<T> *arr, std::size_t count) {
            // Specialized implementation
            std::complex<T> acc[chunk_size];
            std::size_t to_sum = count - count % chunk_size;
            
            // Sum by chunks
            asm volatile ("nop;nop;nop;");
            chunk_sum<T, chunk_size, chunk_size % 2>::compute(acc, arr, to_sum);
            asm volatile ("nop;nop;nop;");

            std::size_t i = to_sum;
            
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
    
    template <class T, std::size_t chunk_size, std::size_t reg_size>
    struct sum {
        static force_inline std::complex<T> compute(std::complex<T> *arr, const std::size_t count) {
            // Default implementation
            std::complex<T> result(0,0);
            
            for (std::size_t i = 0; i < count; i++) {
                result += arr[i];
            }
            
            return result;
        }
    };
    
}

    template<class T, std::size_t chunk_size>
    static std::complex<T> sum(std::complex<T> *arr, std::size_t count) {
        return implementation::sum<T, chunk_size>::compute(arr, count);
    }

    template<class T, std::size_t chunk_size>
    struct ToTest {
        static std::complex<T> to_test(std::complex<T> *arr, std::size_t count) {
            return sum<T, chunk_size>(arr, count);
        }
    };
    
}


