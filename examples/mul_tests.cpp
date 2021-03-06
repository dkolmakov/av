// #pragma GCC optimize("O3", "unroll-loops", "omit-frame-pointer", "inline")
// #pragma GCC option("arch=native", "tune=native", "no-zero-upper")

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <array>
#include <vector>

#include "array_mul.hpp"

#include "mul_simple.hpp"
#include "mul_unroll.hpp"
#include "mul_old.hpp"
#include "mul_man.hpp"

#include "test_harness.hpp"

using namespace av;

typedef KernelParameters<std::size_t, 1, 2, 4, 8, 16, 32> chunk_sizes;
typedef KernelParameters<std::size_t, 1, 2, 4, 8> chunk_numbers;
typedef Kernels<mul_simple::chunk_mul, 
                mul_unroll::chunk_mul, 
                mul_man::chunk_mul, 
                mul_old::chunk_mul> mul_kernels;
typedef Combinations<mul_kernels, chunk_sizes, chunk_numbers> tuples;

typedef TestHarness<array_mul::test_function<double>, tuples::val> array_mul_harness;
typedef array_mul::test_function<double>::input_data array_mul_input;


int main(int argc, char **argv) {
    if (argc < 3)
        return 1;
    
    std::size_t count = atoi(argv[1]);
    std::size_t repeats = atoi(argv[2]);

    Benchmark<array_mul_input>* array_mul_benchmark = array_mul_harness::prepare_benchmark("array_mul");
    
    std::cout << av::inst_set << " instruction set" << std::endl;
    
    for (std::size_t i = 0; i < repeats; i++)
        array_mul_benchmark->run(count);
    
    array_mul_benchmark->print_results();

//     PairsPrinter<mul_kernels_chunk_sizes_numbers::next, mul_kernels_chunk_sizes_numbers::total - 1>::print();

    delete array_mul_benchmark;
    return 0;
}
