// #pragma GCC optimize("O3", "unroll-loops", "omit-frame-pointer", "inline")
// #pragma GCC option("arch=native", "tune=native", "no-zero-upper")

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <array>
#include <vector>

#include "common.hpp"

#include "array_sum.hpp"
#include "sum_simple.hpp"
#include "sum_unroll.hpp"
#include "sum_man.hpp"

#include "test_harness.hpp"

using namespace av;

typedef KernelParameters<std::size_t, 1, 2, 4> chunk_sizes;
typedef KernelParameters<std::size_t, 1, 2, 4, 8> chunk_numbers;
typedef Kernels<sum_simple::chunk_sum, sum_unroll::chunk_sum, sum_man::chunk_sum> sum_kernels;
typedef Combinations<sum_kernels, chunk_sizes, chunk_numbers> tuples;
                
static std::complex<float> init() {return {(float)(std::rand()) / RAND_MAX, (float)(std::rand()) / RAND_MAX};}

typedef array_sum::test_function<std::complex<float>, init> tf;
typedef TestHarness<tf::core, tf::input_data, tuples::val> array_sum_harness;
typedef tf::input_data array_sum_input;

int main(int argc, char **argv) {
    if (argc < 3)
        return 1;
    
    std::size_t count = atoi(argv[1]);
    std::size_t repeats = atoi(argv[2]);
    std::cout << av::inst_set << " instruction set" << std::endl;

    Benchmark<array_sum_input>* array_sum_benchmark = array_sum_harness::prepare_benchmark("array_sum");
    
    for (std::size_t i = 0; i < repeats; i++)
        array_sum_benchmark->run(count);
    
    array_sum_benchmark->print_results();

    delete array_sum_benchmark;
    return 0;
}
