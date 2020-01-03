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
#include "sum_chunked.hpp"
#include "sum_man_sse.hpp"
#include "sum_man_avx.hpp"

#include "test_harness.hpp"

using namespace av_prof;

typedef KernelParameters<1, 2, 4, 8, 16, 32> chunk_sizes;

typedef KernelParameters<1, 2, 4, 8> chunk_numbers;

// typedef Kernels<sum_simple::chunk_sum, 
//                 sum_unroll::chunk_sum, 
//                 sum_chunked::chunk_sum,
//                 sum_man_sse::chunk_sum,
//                 sum_man_avx::chunk_sum> sum_kernels;
// typedef TestHarness<array_sum::test_function<double>, sum_kernels, chunk_sizes> array_sum_harness;
// typedef array_sum::test_function<double>::input_data array_sum_input;
// 
// Benchmark<array_sum_input>* array_sum_benchmark = array_sum_harness::prepare_benchmark("array_sum");                


int main(int argc, char **argv) {
    if (argc < 2)
        return 1;
    
    std::size_t count = atoi(argv[1]);
    std::cout << av::inst_set << " instruction set" << std::endl;

//     array_sum_benchmark->run(count);
//     PairsPrinter<mul_kernels_chunk_sizes_numbers::next, mul_kernels_chunk_sizes_numbers::total - 1>::print();

    return 0;
}