#pragma once

#include <complex>
#include <vector>
#include <random>

#include "common.hpp"
#include "test_harness.hpp"

namespace array_mul {

    template<class T> 
    struct test_function {

        struct input_data {
            std::vector<std::complex<T>> first;
            std::vector<std::complex<T>> second;
            std::vector<std::complex<T>> reference;
            
            input_data(std::size_t count) : first(count), second(count), reference(count) {
                for (size_t i = 0; i < count; i++) {
                    first[i] = {(double)(std::rand()) / RAND_MAX, (double)(std::rand()) / RAND_MAX};
                    second[i] = {(double)(std::rand()) / RAND_MAX, (double)(std::rand()) / RAND_MAX};
                    reference[i] = first[i] * second[i];
                }
            }
        };
        
        
        template<class params_tuple>
        struct core {
            typedef typename av_prof::ByIndex<params_tuple, 0>::elem::val chunk_mul;
            static const std::size_t chunk_size = av_prof::ByIndex<params_tuple, 1>::elem::val;
            static const std::size_t n_chunks = av_prof::ByIndex<params_tuple, 2>::elem::val;

            static std::string get_label() {
                return chunk_mul::get_label() + " with " + std::to_string(n_chunks) + " chunks of " + std::to_string(chunk_size);
            }
            
            static bool compute(input_data& input) {
                const std::size_t portion_size = chunk_size * n_chunks;

                std::size_t count = input.first.size();
                std::complex<T> *first = input.first.data();
                std::complex<T> *second = input.second.data();
                std::complex<T> *third = new std::complex<T>[count];
                
                const std::size_t to_sum = count - count % portion_size;
                
                std::complex<T> *right[n_chunks];
                std::complex<T> *left[n_chunks];
                std::complex<T> *res[n_chunks];
                
                asm volatile ("nop;nop;nop;");
                for (std::size_t i = 0; i < to_sum; i += portion_size) {
                    
                    for (std::size_t j = 0; j < n_chunks; j++) {
                        right[j] = first + i + j * chunk_size;
                        left[j] = second + i + j * chunk_size;
                        res[j] = third + i + j * chunk_size;
                    }
                    chunk_mul::template core<T, chunk_size, n_chunks>::compute(res, left, right);
                }
                asm volatile ("nop;nop;nop;");

                // Handle the remainder
                for (std::size_t i = to_sum; i < count; i++) {
                    third[i] = first[i] * second[i];
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


