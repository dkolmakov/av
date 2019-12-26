#pragma once

#include <complex>

#include "common.hpp"

namespace array_sum {

    struct test_function {
        
        template<class T, std::size_t chunk_size, template<class TT, std::size_t sz> class chunk_sum>
        struct core {
            static std::complex<T> compute(std::complex<T> *arr, std::size_t count) {
                std::complex<T> acc[chunk_size];
                const std::size_t to_sum = count - count % chunk_size;
                
                for (std::size_t i = 0; i < chunk_size; i++)
                    acc[i] = 0;
                
                // Sum by chunks
                asm volatile ("nop;nop;nop;");
                for (std::size_t i = 0; i < to_sum; i += chunk_size) {
                    chunk_sum<T, chunk_size>::compute(acc, arr + i);
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
        };
        
        template<class T>
        struct input_data {
            std::vector<std::complex<T>> arr;
            
            input_data(std::size_t count) : arr(count) {
                for (size_t i = 0; i < count; i++)
                    arr[i] = i + 1;
            }
            
            std::complex<T> get_reference() {
                std::complex<T> result = 0;
                
                for (std::size_t i = 0; i < arr.size(); i++) {
                    result += arr[i];
                }

                return result;
            }
        };
        
    };
}


