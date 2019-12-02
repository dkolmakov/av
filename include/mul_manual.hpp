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
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr, std::size_t count) {
            // Default implementation
            std::complex<T> res(1,0);
            for (std::size_t i = 0; i < count; i++) {
                res *= arr[i];
            }
            *acc = res;
        }
    };
    
    template <class T>
    struct chunk_mul<T, 1> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr, std::size_t count) {
            __m128d res = _mm_load_pd(reinterpret_cast<double *>(arr));
            
            for (std::size_t i = 1; i < count; i += 1) {
                __m128d v0 = _mm_loadu_pd(reinterpret_cast<double *>(arr + i));
                
                __m128d tmp0 = _mm_mul_pd(res, v0);
//                 v0 = _mm_permute_pd(v0, 0x0);
                v0 = _mm_shuffle_pd(v0, v0, 1);
                
                __m128d odd_signbits = _mm_setr_pd(0, -0.0);
                res = _mm_xor_pd(res, odd_signbits);
                
                __m128d tmp1 = _mm_mul_pd(res, v0);
                
                res = _mm_addsub_pd(tmp0, tmp1);
            }
            
            _mm_store_pd(reinterpret_cast<double *>(acc), res);
        }
    };

    template <class T>
    struct chunk_mul<T, 2> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr, std::size_t count) {
            __m256d res = _mm256_setr_pd(1.0, 0, 1.0, 0);
            
            for (std::size_t i = 0; i < count; i += 2) {
                __m256d vec2 = _mm256_loadu_pd(reinterpret_cast<double *>(arr + i));
                
                /* Step 1: Multiply res and vec2 */
                __m256d vec3 = _mm256_mul_pd(res, vec2);

                /* Step 2: Switch the real and imaginary elements of vec2 */
                vec2 = _mm256_permute_pd(vec2, 0x5);    // gcc turns this into the slower lane-crossing VPERMPD, not VPERMILPD, but clang is ok.
                
                /* Step 3: Negate the imaginary elements of vec2 */
                    //  vec2 = _mm256_mul_pd(vec2, neg);       // this is much slower than XOR
                // Flipping the sign bit in vec1 lets this run in parallel with the shuffle on vec2, reducing latency
                //__m256d odd_signbits = _mm256_castsi256_pd( _mm256_setr_epi64x(0, 1ULL<<63, 0, 1ULL<<63));
                __m256d odd_signbits = _mm256_setr_pd(0, -0.0, 0, -0.0);
                res = _mm256_xor_pd(res, odd_signbits);
                
                /* Step 4: Multiply vec1 and the modified vec2 */
                __m256d vec4 = _mm256_mul_pd(res, vec2);

                /* Horizontally subtract the elements in vec3 and vec4 */
                res = _mm256_hsub_pd(vec3, vec4);
            }
            
            _mm256_storeu_pd((double*) acc, res);
        }
    };

    template <class T>
    struct chunk_mul<T, 4> {
        static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr, std::size_t count) {
            std::cout << "No solution for AVX512 yet" << std::endl;
            chunk_mul<T, 0>::compute(acc, arr, count);
            acc[1] = {1,0};
            acc[2] = {1,0};
            acc[3] = {1,0};
        }
    };

    template <class T, std::size_t chunk_size = av::SIMD_REG_SIZE / sizeof(T) / 2>
    struct mul;
    
    template <class T>
    struct mul<T, 0> {
        static force_inline std::complex<T> compute(std::complex<T> *arr, const std::size_t count) {
            // Default implementation
            std::complex<T> acc(1,0);
            
            asm volatile ("nop;nop;nop;");
            chunk_mul<T, 0>::compute(&acc, arr, count);
            asm volatile ("nop;nop;nop;");
            
            return acc;
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


