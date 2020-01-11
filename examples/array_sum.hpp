#pragma once

#include <complex>

#include "common.hpp"
#include "test_harness.hpp"

namespace array_sum {

    template<class T> 
    struct test_function {

        struct input_data {
            std::vector<std::complex<T>> arr;
            std::complex<T> reference;
            
            input_data(std::size_t count) : arr(count), reference(0) {
                for (size_t i = 0; i < count; i++) {
                    arr[i] = i + 1;
                    reference += arr[i];
                }
            }
        };
        
        
        template<class params_tuple>
        struct core {
            typedef typename av_prof::ByIndex<params_tuple, 0>::elem::val chunk_sum;
            static const std::size_t chunk_size = av_prof::ByIndex<params_tuple, 1>::elem::val;
            static const std::size_t n_chunks = av_prof::ByIndex<params_tuple, 2>::elem::val;

            static std::string get_label() {
                return chunk_sum::get_label() + " with " + std::to_string(n_chunks) + " chunks of " + std::to_string(chunk_size);
            }
            
            static bool compute(input_data& input) {
                const std::size_t portion_size = chunk_size * n_chunks;

                std::size_t count = input.arr.size();
                std::complex<T> *arr = input.arr.data();
                
                std::complex<T> acc[portion_size];
                const std::size_t to_sum = count - count % portion_size;
                
                for (std::size_t i = 0; i < portion_size; i++)
                    acc[i] = 0;
                
                std::complex<T> *left[n_chunks];
                for (std::size_t j = 0; j < n_chunks; j++)
                    left[j] = acc + j * chunk_size;

                std::complex<T> *right[n_chunks];
                
                // Sum by chunks
                asm volatile ("nop;nop;nop;");
                for (std::size_t i = 0; i < to_sum; i += portion_size) {
                    for (std::size_t j = 0; j < n_chunks; j++)
                        right[j] = arr + i + j * chunk_size;

                    chunk_sum::template core<T, chunk_size, n_chunks>::compute(left, left, right);
                }
                asm volatile ("nop;nop;nop;");
                
                // Add the remainder
                std::complex<T> result(0,0);
                
                for (std::size_t i = to_sum; i < count; i++) {
                    result += arr[i];
                }
                
                for (std::size_t j = 0; j < portion_size; j++) {
                    result += acc[j];
                }
                
                return abs(result - input.reference) < 1e-6;
            }
        };
        
    };
}


