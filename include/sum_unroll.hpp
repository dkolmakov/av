#pragma once

#include <complex>

#include "common.hpp"

namespace sum_unroll {
    
namespace implementation {

    template <class T, std::size_t index>
    struct chunk_sum;

    template <class T>
    struct chunk_sum<T, 0> {
        static force_inline void compute(T *acc, T *arr) {
            acc[0] += arr[0];
        }
    };

    template <class T, std::size_t index>
    struct chunk_sum {
        static force_inline void compute(T *acc, T *arr) {
            chunk_sum<T, index - 1>::compute(acc, arr);
            acc[index] += arr[index];
        }
    };
}

    struct chunk_sum {
        static std::string get_label() {
            return "sum_unroll";
        }
        
        template <class T, std::size_t chunk_size>
        struct core {
            static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr) {
                return implementation::chunk_sum<T, chunk_size * 2 - 1>::compute((T *)acc, (T *)arr);
            }
        };
    };
    
    template<class T, std::size_t chunk_size>
    static std::complex<T> sum(std::complex<T> *arr, std::size_t count) {
        std::complex<T> acc[chunk_size];
        const std::size_t to_sum = count - count % chunk_size;
        
        for (std::size_t i = 0; i < chunk_size; i++)
            acc[i] = 0;
        
        // Sum by chunks
        asm volatile ("nop;nop;nop;");
        for (std::size_t i = 0; i < to_sum; i += chunk_size) {
            chunk_sum::core<T, chunk_size>::compute(acc, arr + i);
        }
        asm volatile ("nop;nop;nop;");
        
        // Add the remainder
        std::complex<T> result(0,0);
        std::size_t j = 0;
        for (std::size_t i = to_sum; i < count; i++, j++) {
            result += arr[i] + acc[j];
        }
        for (; j < chunk_size; j++) {
            result += acc[j];
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


