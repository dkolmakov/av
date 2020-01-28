#pragma once

#include <complex>
#include <vector>
#include <random>

#include "common.hpp"

namespace array_sum {

    template<class T, class Treg, T init_value(void)> 
    struct test_function {

        struct input_data {
            std::vector<T> first;
            std::vector<T> second;
            std::vector<T> reference;
            
            input_data(std::size_t count) : first(count), second(count), reference(count) {
                for (size_t i = 0; i < count; i++) {
                    first[i] = init_value();
                    second[i] = init_value();
                    reference[i] = first[i] + second[i];
                }
            }
        };
        
        template<class params_tuple>
        struct core {
            typedef typename params_tuple::template ByIndex<0>::elem::val chunk_sum;
            static const std::size_t chunk_size = params_tuple::template ByIndex<1>::elem::val;
            static const std::size_t n_chunks = params_tuple::template ByIndex<2>::elem::val;

            static std::string get_label() {
                return chunk_sum::get_label() + " with " + std::to_string(n_chunks) + " chunks of " + std::to_string(chunk_size);
            }

            static bool compute(input_data& input) {
                const std::size_t portion_size = chunk_size * n_chunks;

                std::size_t count = input.first.size();
                T *first = input.first.data();
                T *second = input.second.data();
                T *third = new T[count];
                
                const std::size_t to_sum = count - count % portion_size;
                
                T *left[n_chunks];
                T *right[n_chunks];
                T *res[n_chunks];
                
                // Sum by chunks
                asm volatile ("nop;nop;nop;");
                for (std::size_t i = 0; i < to_sum; i += portion_size) {
                    for (std::size_t j = 0; j < n_chunks; j++) {
                        right[j] = first + i + j * chunk_size;
                        left[j] = second + i + j * chunk_size;
                        res[j] = third + i + j * chunk_size;
                    }

                    chunk_sum::template core<T, Treg, chunk_size, n_chunks>::compute(res, left, right);
                }
                asm volatile ("nop;nop;nop;");
                
                // Add the remainder
                for (std::size_t i = to_sum; i < count; i++) {
                    third[i] = first[i] + second[i];
                }

                bool result = true;
                for (size_t i = 0; i < count; i++) {
                    result = result && (abs(third[i] - input.reference[i]) < 1e-6);
                }
                
                delete[] third;
                return result;
            }
        };
        
    };
}


