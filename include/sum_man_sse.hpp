#pragma once

#include <immintrin.h>
#include <complex>

#include "common.hpp"

namespace sum_man_sse {
    
namespace implementation {

    template <class T, std::size_t index>
    struct init;

    template <class T>
    struct init<T, 0> {
        static force_inline void doIt(__m128d *acc) {
            acc[2 * 0] = _mm_setr_pd(0, 0);
            acc[2 * 0 + 1] = _mm_setr_pd(0, 0);
        }
    };

    template <class T, std::size_t index>
    struct init {
        static force_inline void doIt(__m128d *acc) {
            acc[2 * index] = _mm_setr_pd(0, 0);
            acc[2 * index + 1] = _mm_setr_pd(0, 0);
            init<T, index - 1>::doIt(acc);
        }
    };
    
    template <class T, std::size_t index>
    struct unpack;

    template <class T>
    struct unpack<T, 0> {
        static force_inline void doIt(__m128d *vals, std::complex<T> *arr) {
            vals[2 * 0] = _mm_loadu_pd((double*)(arr + 2 * 0));
            vals[2 * 0 + 1] = _mm_loadu_pd((double*)(arr + 2 * 0 + 1));
        }
    };

    template <class T, std::size_t index>
    struct unpack {
        static force_inline void doIt(__m128d *vals, std::complex<T> *arr) {
            vals[2 * index] = _mm_loadu_pd((double*)(arr + 2 * index));
            vals[2 * index + 1] = _mm_loadu_pd((double*)(arr + 2 * index + 1));
            unpack<T, index - 1>::doIt(vals, arr);
        }
    };

    template <class T, std::size_t index>
    struct summation;
    
    template <class T>
    struct summation<T, 0> {
        static force_inline void doIt(__m128d *acc, __m128d *vals) {
            acc[2 * 0] = _mm_add_pd(acc[2 * 0], vals[2 * 0]);
            acc[2 * 0 + 1] = _mm_add_pd(acc[2 * 0 + 1], vals[2 * 0 + 1]);
        }
    };
    
    template <class T, std::size_t index>
    struct summation {
        static force_inline void doIt(__m128d *acc, __m128d *vals) {
            acc[2 * index] = _mm_add_pd(acc[2 * index], vals[2 * index]);
            acc[2 * index + 1] = _mm_add_pd(acc[2 * index + 1], vals[2 * index + 1]);

            summation<T, index - 1>::doIt(acc, vals);
        }
    };
    
    template <class T, std::size_t index>
    struct pack;

    template <class T>
    struct pack<T, 0> {
        static force_inline void doIt(std::complex<T> *dst, __m128d *acc) {
            _mm_storeu_pd((double*)(dst + 2 * 0), acc[2 * 0]);
            _mm_storeu_pd((double*)(dst + 2 * 0 + 1), acc[2 * 0 + 1]);
        }
    };

    template <class T, std::size_t index>
    struct pack {
        static force_inline void doIt(std::complex<T> *dst, __m128d *acc) {
            _mm_storeu_pd((double*)(dst + 2 * index), acc[2 * index]);
            _mm_storeu_pd((double*)(dst + 2 * index + 1), acc[2 * index + 1]);
            pack<T, index - 1>::doIt(dst, acc);
        }
    };

    
    template <class T, std::size_t chunk_size, std::size_t parity_checker>
    struct chunk_sum;
    
    template <class T, std::size_t chunk_size>
    struct chunk_sum<T, chunk_size, 0> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr, std::size_t count) {
            __m128d dataA[chunk_size / 2];
            init<T, chunk_size / 2 - 1>::doIt(dataA);
            
            for (std::size_t i = 0; i < count; i += chunk_size) {
                __m128d dataB[chunk_size / 2];
                
                unpack<T, chunk_size / 2 - 1>::doIt(dataB, arr + i);
                summation<T, chunk_size / 2 - 1>::doIt(dataA, dataB);
            }

            pack<T, chunk_size / 2 - 1>::doIt(acc, dataA);
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
    struct sum<T, chunk_size, 16> {
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


