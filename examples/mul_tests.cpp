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
#include "mul_old_sse.hpp"
#include "mul_old_avx.hpp"
#include "mul_man.hpp"

#include "test_harness.hpp"

using namespace av_prof;

typedef KernelParameters<1, 2, 4, 8, 16, 32> chunk_sizes;
typedef KernelParameters<1, 2, 4, 8> chunk_numbers;
typedef Kernels<mul_simple::chunk_mul, mul_unroll::chunk_mul, mul_man::chunk_mul> mul_kernels;

typedef Pairs<mul_kernels, chunk_sizes> mul_kernels_chunk_sizes;
typedef Pairs<mul_kernels_chunk_sizes, chunk_numbers> mul_kernels_chunk_sizes_numbers;

typedef TestHarness<array_mul::test_function<double>, mul_kernels_chunk_sizes_numbers> array_mul_harness;
typedef array_mul::test_function<double>::input_data array_mul_input;


int main(int argc, char **argv) {
    if (argc < 2)
        return 1;
    
    std::size_t count = atoi(argv[1]);
    std::cout << av::inst_set << " instruction set" << std::endl;

    Benchmark<array_mul_input>* array_mul_benchmark = array_mul_harness::prepare_benchmark("array_mul");
    
    array_mul_benchmark->run(count);
    array_mul_benchmark->print_results();

//     PairsPrinter<mul_kernels_chunk_sizes_numbers::next, mul_kernels_chunk_sizes_numbers::total - 1>::print();

    delete array_mul_benchmark;
    return 0;
}
