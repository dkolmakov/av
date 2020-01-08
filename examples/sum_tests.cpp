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

using namespace av_prof;

typedef KernelParameters<1, 2, 4, 8, 16, 32> chunk_sizes;
typedef KernelParameters<1, 2, 4, 8> chunk_numbers;
typedef Kernels<sum_simple::chunk_sum, sum_unroll::chunk_sum, sum_man::chunk_sum> sum_kernels;
                
typedef Pairs<sum_kernels, chunk_sizes> sum_kernels_chunk_sizes;
typedef Pairs<sum_kernels_chunk_sizes, chunk_numbers> sum_kernels_chunk_sizes_numbers;
                
typedef TestHarness<array_sum::test_function<double>, sum_kernels_chunk_sizes_numbers> array_sum_harness;
typedef array_sum::test_function<double>::input_data array_sum_input;

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
