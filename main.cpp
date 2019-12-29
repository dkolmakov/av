// #pragma GCC optimize("O3", "unroll-loops", "omit-frame-pointer", "inline")
// #pragma GCC option("arch=native", "tune=native", "no-zero-upper")

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <array>
#include <vector>

#include "common.hpp"

#include "array_sum.hpp"
#include "array_mul.hpp"

#include "sum_simple.hpp"
#include "sum_unroll.hpp"
#include "sum_chunked.hpp"
#include "sum_man_sse.hpp"
#include "sum_man_avx.hpp"

#include "mul_simple.hpp"
#include "mul_unroll.hpp"
#include "mul_old_sse.hpp"
#include "mul_old_avx.hpp"
#include "mul_avx.hpp"
#include "mul_sse.hpp"

#include "test_harness.hpp"

typedef KernelParameters<1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64> chunk_sizes;

typedef Kernels<sum_simple::chunk_sum, 
                sum_unroll::chunk_sum, 
                sum_chunked::chunk_sum,
                sum_man_sse::chunk_sum,
                sum_man_avx::chunk_sum> sum_kernels;
typedef TestHarness<array_sum::test_function<double>, sum_kernels, chunk_sizes> array_sum_harness;
typedef array_sum::test_function<double>::input_data array_sum_input;

Benchmark<array_sum_input>* array_sum_benchmark = array_sum_harness::prepare_benchmark("array_sum");                

typedef Kernels<mul_simple::chunk_mul, 
                mul_unroll::chunk_mul, 
                mul_old_sse::chunk_mul, 
                mul_old_avx::chunk_mul, 
                mul_sse::chunk_mul,
                mul_avx::chunk_mul> mul_kernels;
typedef TestHarness<array_mul::test_function<double>, mul_kernels, chunk_sizes> array_mul_harness;
typedef array_mul::test_function<double>::input_data array_mul_input;

Benchmark<array_mul_input>* array_mul_benchmark = array_mul_harness::prepare_benchmark("array_mul");                


int main(int argc, char **argv) {
    if (argc < 2)
        return 1;
    
    std::size_t count = atoi(argv[1]);

    std::cout << av::inst_set << " instruction set" << std::endl;

    array_sum_benchmark->run(count);
    array_mul_benchmark->run(count);

    return 0;
}
