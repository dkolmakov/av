#pragma once

#include <immintrin.h>
#include <complex>

#include "common.hpp"

namespace mul_old_avx {
    
namespace implementation {

    template <class T, std::size_t index>
    struct unpack;

    template <class T>
    struct unpack<T, 0> {
        static force_inline void doIt(__m256d *vals, std::complex<T> *arr) {
            vals[0] = _mm256_loadu_pd((double*)(arr + 0));
        }
    };

    template <class T, std::size_t index>
    struct unpack {
        static force_inline void doIt(__m256d *vals, std::complex<T> *arr) {
            vals[index] = _mm256_loadu_pd((double*)(arr + 2 * index));
            unpack<T, index - 1>::doIt(vals, arr);
        }
    };

    template <class T, std::size_t index>
    struct multiply;
    
    template <class T>
    struct multiply<T, 0> {
        static force_inline void doIt(__m256d *acc, __m256d *vals) {
            /* Step 1: Multiply res and vec2 */
            __m256d vec0 = _mm256_mul_pd(acc[0], vals[0]);

            /* Step 2: Switch the real and imaginary elements of vec2 */
            vals[0] = _mm256_permute_pd(vals[0], 0x5); 
            
            /* Step 3: Negate the imaginary elements of vec2 */
                //  vec2 = _mm256_mul_pd(vec2, neg);       // this is much slower than XOR
            // Flipping the sign bit in vec1 lets this run in parallel with the shuffle on vec2, reducing latency
            //__m256d odd_signbits = _mm256_castsi256_pd( _mm256_setr_epi64x(0, 1ULL<<63, 0, 1ULL<<63));
            __m256d odd_signbits = _mm256_setr_pd(0, -0.0, 0, -0.0);
            acc[0] = _mm256_xor_pd(acc[0], odd_signbits);
            
            /* Step 4: Multiply vec1 and the modified vec2 */
            __m256d vec1 = _mm256_mul_pd(acc[0], vals[0]);

            /* Horizontally subtract the elements in vec3 and vec4 */
            acc[0] = _mm256_hsub_pd(vec0, vec1);
        }
    };
    
    // Original version id from https://www.codeproject.com/Articles/874396/Crunching-Numbers-with-AVX-and-AVX
    // And improved one is here https://stackoverflow.com/questions/39509746/how-to-square-two-complex-doubles-with-256-bit-avx-vectors

    template <class T, std::size_t index>
    struct multiply {
        static force_inline void doIt(__m256d *acc, __m256d *vals) {
            /* Step 1: Multiply res and vec2 */
            __m256d vec0 = _mm256_mul_pd(acc[index], vals[index]);

            /* Step 2: Switch the real and imaginary elements of vec2 */
            vals[index] = _mm256_permute_pd(vals[index], 0x5); 
            
            /* Step 3: Negate the imaginary elements of vec2 */
                //  vec2 = _mm256_mul_pd(vec2, neg);       // this is much slower than XOR
            // Flipping the sign bit in vec1 lets this run in parallel with the shuffle on vec2, reducing latency
            //__m256d odd_signbits = _mm256_castsi256_pd( _mm256_setr_epi64x(0, 1ULL<<63, 0, 1ULL<<63));
            __m256d odd_signbits = _mm256_setr_pd(0, -0.0, 0, -0.0);
            acc[index] = _mm256_xor_pd(acc[index], odd_signbits);
            
            /* Step 4: Multiply vec1 and the modified vec2 */
            __m256d vec1 = _mm256_mul_pd(acc[index], vals[index]);

            /* Horizontally subtract the elements in vec3 and vec4 */
            acc[index] = _mm256_hsub_pd(vec0, vec1);

            multiply<T, index - 1>::doIt(acc, vals);
        }
    };
    
    template <class T, std::size_t index>
    struct pack;

    template <class T>
    struct pack<T, 0> {
        static force_inline void doIt(std::complex<T> *dst, __m256d *acc) {
            _mm256_storeu_pd((double*)(dst + 2 * 0), acc[0]);
        }
    };

    template <class T, std::size_t index>
    struct pack {
        static force_inline void doIt(std::complex<T> *dst, __m256d *acc) {
            _mm256_storeu_pd((double*)(dst + 2 * index), acc[index]);
            pack<T, index - 1>::doIt(dst, acc);
        }
    };
    
    template <class T, std::size_t chunk_size, std::size_t parity_checker>
    struct chunk_mul;
    
    template <class T, std::size_t chunk_size>
    struct chunk_mul<T, chunk_size, 0> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr, std::size_t count) {
            __m256d res[chunk_size / 2];
            unpack<T, chunk_size / 2 - 1>::doIt(res, arr);
            
            for (std::size_t i = chunk_size; i < count; i += chunk_size) {
                __m256d v0[chunk_size / 2];
                unpack<T, chunk_size / 2 - 1>::doIt(v0, arr + i);
                multiply<T, chunk_size / 2 - 1>::doIt(res, v0);
            }
            
            pack<T, chunk_size / 2 - 1>::doIt(acc, res);
        }
    };
    
    template <class T, std::size_t chunk_size, std::size_t parity_checker>
    struct chunk_mul {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr, std::size_t count) {
            // Default implementation
            for (std::size_t i = 0; i < chunk_size; i++)
                acc[i] = 1;
            
            std::complex<T> res(1,0);
            for (std::size_t i = 0; i < count; i++) {
                res *= arr[i];
            }
            *acc = res;
        }
    };
    
    template <class T, std::size_t chunk_size, std::size_t reg_size = av::SIMD_REG_SIZE>
    struct mul;
    
    template <class T, std::size_t chunk_size>
    struct mul<T, chunk_size, 32> {
        static force_inline std::complex<T> compute(std::complex<T> *arr, const std::size_t count) {
            // Specialized implementation
            std::complex<T> acc[chunk_size];
            std::size_t to_mul = count - count % chunk_size;
            
            // Sum by chunks
            asm volatile ("nop;nop;nop;");
            chunk_mul<T, chunk_size, chunk_size % 2>::compute(acc, arr, to_mul);
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
    
    template <class T, std::size_t chunk_size, std::size_t reg_size>
    struct mul {
        static force_inline std::complex<T> compute(std::complex<T> *arr, std::size_t count) {
            // Default implementation
            std::complex<T> res(1,0);
            chunk_mul<T, 1, 1>::compute(&res, arr, count);
            return res;
        }
    };
    
}

    template<class T, std::size_t chunk_size>
    static std::complex<T> mul(std::complex<T> *arr, std::size_t count) {
        return implementation::mul<T, chunk_size>::compute(arr, count);
    }

    template<class T, std::size_t chunk_size>
    struct ToTest {
        static std::complex<T> to_test(std::complex<T> *arr, std::size_t count) {
            return mul<T, chunk_size>(arr, count);
        }
    };
    
}


