#pragma once

#include <complex>
#include <vector>

#include "common.hpp"

namespace array_mul {

    template<class T> 
    struct test_function {

        struct input_data {
            std::vector<std::complex<T>> arr;
            std::complex<T> reference;
            
            input_data(std::size_t count) : arr(count), reference(1) {
                for (size_t i = 0; i < count; i++) {
                    arr[i] = 1;
                    if ((i % (size_t)(0.1 * count)) == 0) {
                        arr[i] = 1 + i / (size_t)(0.1 * count);
                    }
                    reference *= arr[i];
                }
            }
        };
        
        
        template<std::size_t chunk_size, template<class TT, std::size_t sz> class chunk_mul>
        struct core {
            static bool compute(input_data& input) {
                std::size_t count = input.arr.size();
                std::complex<T> *arr = input.arr.data();
                
                std::complex<T> acc[chunk_size];
                const std::size_t to_sum = count - count % chunk_size;
                
                for (std::size_t i = 0; i < chunk_size; i++)
                    acc[i] = 1;
                
                // Sum by chunks
                asm volatile ("nop;nop;nop;");
                for (std::size_t i = 0; i < to_sum; i += chunk_size) {
                    chunk_mul<T, chunk_size>::compute(acc, arr + i);
                }
                asm volatile ("nop;nop;nop;");
                
                // Handle the remainder
                std::complex<T> result = 1;
                std::size_t j = 0;
                for (std::size_t i = to_sum; i < count; i++, j++) {
                    result *= arr[i] * acc[j];
                }
                for (; j < chunk_size; j++) {
                    result *= acc[j];
                }
                
                return abs(result - input.reference) < 1e-6;
            }
        };
        
    };
}


