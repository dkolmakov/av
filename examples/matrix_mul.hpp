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
            
            
            static std::complex<T> multiply_and_sum(std::complex<T> *acc, std::complex<T> *first, std::complex<T> *second, std::size_t count) {
                const std::size_t portion_size = chunk_size * n_chunks;
                const std::size_t to_calc = count - count % portion_size;
                
                for (std::size_t i = 0; i < portion_size; i++)
                    mul_acc[i] = 1;

                for (std::size_t i = 0; i < portion_size; i++)
                    mul_acc[i] = 1;
                
                std::complex<T> *left[n_chunks];
                for (std::size_t j = 0; j < n_chunks; j++)
                    left[j] = acc + j * chunk_size;

                std::complex<T> *right[n_chunks];
                
                asm volatile ("nop;nop;nop;");
                for (std::size_t i = 0; i < to_sum; i += portion_size) {
                    
                    for (std::size_t j = 0; j < n_chunks; j++)
                        right[j] = arr + i + j * chunk_size;

                    chunk_mul::template core<T, chunk_size, n_chunks>::compute(left, right);
                }
                asm volatile ("nop;nop;nop;");
                
                // Handle the remainder
                std::complex<T> result = 1;

                for (std::size_t i = to_sum; i < count; i++) {
                    result *= arr[i];
                }

                for (std::size_t j = 0; j < portion_size; j++) {
                    result *= acc[j];
                }
                
            }
            
            static bool compute(input_data& input) {

                const std::size_t portion_size = chunk_size * n_chunks;

                std::size_t count = input.count;
                std::complex<T> *first = input.first.data();
                std::complex<T> *second = input.first.data();
                std::complex<T> *third = new std::complex<T>[count * count];

                // TODO
//                 for (size_t k = 0; k < count; k++) {
//                     reference[i * count + j] += first[i * count + k] * second[k * count + j];
//                 }
                
                
                
                
                bool result = true;
                for (size_t i = 0; i < count * count; i++) {
                    result = result && (abs(third[i] - input.reference[i]) < 1e-6);
                }
                return result;
            }
        };
    };
}


