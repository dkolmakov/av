#pragma once

#include <complex>
#include <vector>
#include <random>

#include "common.hpp"

namespace matrix_mul {

    template<class T> 
    struct test_function {

        struct input_data {
            std::size_t count;
            std::vector<std::complex<T>> first;
            std::vector<std::complex<T>> second;
            
            std::vector<std::complex<T>> reference;
            
            input_data(std::size_t _count) : count(_count), first(count * count), second(count * count), reference(count * count) {
                for (size_t i = 0; i < count * count; i++) {
                    first[i] = {(double)(std::rand()) / RAND_MAX, (double)(std::rand()) / RAND_MAX};
                    second[i] = {(double)(std::rand()) / RAND_MAX, (double)(std::rand()) / RAND_MAX};
                }
                
                for (size_t i = 0; i < count; i++) {
                    for (size_t j = 0; j < count; j++) {
                        for (size_t k = 0; k < count; k++) {
                            reference[i * count + j] += first[i * count + k] * second[k * count + j];
                        }
                    }
                }
            }
        };
        
        
        template<class params_tuple>
        struct core {
            typedef typename params_tuple::left::left::left::val chunk_mul;
            typedef typename params_tuple::left::left::right::val chunk_sum;
            static const std::size_t chunk_size = params_tuple::left::right::val;
            static const std::size_t n_chunks = params_tuple::right::val;

            static std::string get_label() {
                return chunk_mul::get_label() + "/" + chunk_sum::get_label() + " with " + std::to_string(n_chunks) + " chunks of " + std::to_string(chunk_size);
            }
            
            static bool compute(input_data& input) {

                const std::size_t portion_size = chunk_size * n_chunks;

                std::size_t count = input.count;
                std::complex<T> *first = input.first.data();
                std::complex<T> *second = input.first.data();
                std::complex<T> *third = new std::complex<T>[count * count];

                // TODO
                
                bool result = true;
                for (size_t i = 0; i < count * count; i++) {
                    result = result && (abs(third[i] - input.reference[i]) < 1e-6);
                }
                return result;
            }
        };
    };
}


