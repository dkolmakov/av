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
//                     first[i] = {(double)(std::rand()) / RAND_MAX, (double)(std::rand()) / RAND_MAX};
//                     second[i] = {(double)(std::rand()) / RAND_MAX, (double)(std::rand()) / RAND_MAX};
                    first[i] = 1;
                    second[i] = i % count;
                }
                
                for (size_t i = 0; i < count; i++) {
                    for (size_t j = 0; j < count; j++) {
                        for (size_t k = 0; k < count; k++) {
                            reference[i * count + j] += first[i * count + k] * second[j * count + k];
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
            static const std::size_t portion_size = chunk_size * n_chunks;

            static std::string get_label() {
                return chunk_mul::get_label() + "/" + chunk_sum::get_label() + " with " + std::to_string(n_chunks) + " chunks of " + std::to_string(chunk_size);
            }
            
            static std::complex<T> multiply_and_sum(std::complex<T> *acc, std::complex<T> *first, std::complex<T> *second, std::size_t count) {
                const std::size_t to_calc = count - count % portion_size;
                
                std::complex<T> *mul_acc[n_chunks];
                for (std::size_t j = 0; j < n_chunks; j++)
                    mul_acc[j] = acc + j * chunk_size;

                std::complex<T> *sum_acc[n_chunks];
                for (std::size_t j = 0; j < n_chunks; j++)
                    sum_acc[j] = acc + portion_size + j * chunk_size;
            
                for (std::size_t i = 0; i < portion_size; i++)
                    acc[portion_size + i] = 0;

                std::complex<T> *mul_left[n_chunks];
                std::complex<T> *mul_right[n_chunks];
                
                asm volatile ("nop;nop;nop;");
                for (std::size_t i = 0; i < to_calc; i += portion_size) {
                    
                    for (std::size_t j = 0; j < n_chunks; j++) {
                        mul_left[j] = first + i + j * chunk_size;
                        mul_right[j] = second + i + j * chunk_size;
                    }
                    
                    chunk_mul::template core<T, chunk_size, n_chunks>::compute(mul_acc, mul_left, mul_right);

//                     for (std::size_t j = 0; j < n_chunks; j++) {
//                         for (std::size_t k = 0; k < chunk_size; k++) {
//                             std::cout << mul_right[j][k] << " ";
//                         }
//                     }
//                     std::cout << std::endl;

                    chunk_sum::template core<T, chunk_size, n_chunks>::compute(sum_acc, sum_acc, mul_acc);

//                     for (std::size_t j = 0; j < n_chunks; j++) {
//                         for (std::size_t k = 0; k < chunk_size; k++) {
//                             std::cout << sum_acc[j][k] << " ";
//                         }
//                     }
//                     std::cout << std::endl;
                }
                asm volatile ("nop;nop;nop;");
                
                // Handle the remainder
                std::complex<T> result = 0;

                for (std::size_t j = 0; j < portion_size; j++) {
                    result += acc[portion_size + j];
                }
                
                for (std::size_t i = to_calc; i < count; i++) {
                    result += first[i] * second[i];
                }
                
                return result;
            }
            
            static bool compute(input_data& input) {
                std::size_t count = input.count;
                std::complex<T> *first = input.first.data();
                std::complex<T> *second = input.second.data();
                std::complex<T> *third = new std::complex<T>[count * count];
                std::complex<T> *acc = new std::complex<T>[portion_size * 2];

                for (size_t i = 0; i < count; i++) {
                    for (size_t j = 0; j < count; j++) {
                        third[i * count + j] = multiply_and_sum(acc, first + i * count, second + j * count, count);
                    }
                }
                
                bool result = true;
//                 std::cout << " Result " << std::endl;
                for (size_t i = 0; i < count * count; i++) {
//                     std::cout << third[i] << "(" << input.reference[i] << ") ";
//                     if ((i + 1) % count == 0)
//                         std::cout << std::endl;
                    result = result && (abs(third[i] - input.reference[i]) < 1e-6);
                }
                return result;
            }
        };
    };
}


