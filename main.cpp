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
typedef TestHarness<double, array_sum::test_function, sum_kernels, chunk_sizes> array_sum_harness;

Benchmark<double>* array_sum_benchmark = array_sum_harness::prepare_benchmark("array_sum");                



// std::vector<BenchmarkWrapper<double>*> mul_tasks = {
//     Tests<double, mul_simple::ToTest, chunks>::prepare_benchmarks("mul_simple"),
//     Tests<double, mul_unroll::ToTest, chunks>::prepare_benchmarks("musuml_unroll"),
//     Tests<double, mul_old_sse::ToTest, chunks>::prepare_benchmarks("mul_old_sse"),
//     Tests<double, mul_old_avx::ToTest, chunks>::prepare_benchmarks("mul_old_avx"),
//     Tests<double, mul_sse::ToTest, chunks>::prepare_benchmarks("mul_sse\t"),
//     Tests<double, mul_avx::ToTest, chunks>::prepare_benchmarks("mul_avx\t")
// };

int main(int argc, char **argv) {
    if (argc < 2)
        return 1;
    
    std::size_t count = atoi(argv[1]);

    std::cout << av::inst_set << " instruction set" << std::endl;
    array_sum_harness::run_benchmark(array_sum_benchmark, count);

//     std::complex<double> *arr_to_mul = new std::complex<double>[to_sum];
//     for (size_t i = 0; i < to_sum; i++) {
//         arr_to_mul[i] = 1;
//         if ((i % (size_t)(0.1 * to_sum)) == 0) {
//             arr_to_mul[i] = 1 + i / (size_t)(0.1 * to_sum);
//         }
//     }
//     std::complex<double> mul = mul_simple::mul(arr_to_mul, to_sum);
// 
//     std::cout << "Multiplication: " << av::inst_set << " instruction set" << std::endl;
//     run_benchmarks(mul_tasks, arr_to_mul, to_sum, mul);
   
    return 0;
}
