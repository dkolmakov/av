#pragma once

#include <complex>

#include "common.hpp"

namespace sum_simple {
    
    struct chunk_sum {
        static std::string get_label() {
            return "sum_simple";
        }
        
        template <class T, std::size_t chunk_size>
        struct core {
            static force_inline void compute(std::complex<T> *acc, std::complex<T> *arr) {
                for (std::size_t i = 0; i < chunk_size; i++) {
                    acc[i] += arr[i];
                }
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


