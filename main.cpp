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

typedef KernelParameters<1, 2, 4, 8, 16, 24, 32, 48, 64> chunk_sizes;
typedef Kernels<array_sum::sum,
                sum_simple::chunk_sum, 
                sum_unroll::chunk_sum, 
                sum_chunked::chunk_sum,
                sum_man_sse::chunk_sum,
                sum_man_avx::chunk_sum> sum_kernels;

Benchmark<double>* array_sum_benchmark = TestHarness<double, sum_kernels, chunk_sizes>::prepare_benchmark("array_sum");                

// std::vector<BenchmarkWrapper<double>*> sum_tasks = {
//     Tests<double, sum_simple::ToTest, chunks>::prepare_benchmarks("sum_simple"),
//     Tests<double, sum_unroll::ToTest, chunks>::prepare_benchmarks("sum_unroll"),
//     Tests<double, sum_chunked::ToTest, chunks>::prepare_benchmarks("sum_chunked"),
//     Tests<double, sum_man_sse::ToTest, chunks>::prepare_benchmarks("sum_man_sse"),
//     Tests<double, sum_man_avx::ToTest, chunks>::prepare_benchmarks("sum_man_avx"),
// };
// 
// std::vector<BenchmarkWrapper<double>*> mul_tasks = {
//     Tests<double, mul_simple::ToTest, chunks>::prepare_benchmarks("mul_simple"),
//     Tests<double, mul_unroll::ToTest, chunks>::prepare_benchmarks("mul_unroll"),
//     Tests<double, mul_old_sse::ToTest, chunks>::prepare_benchmarks("mul_old_sse"),
//     Tests<double, mul_old_avx::ToTest, chunks>::prepare_benchmarks("mul_old_avx"),
//     Tests<double, mul_sse::ToTest, chunks>::prepare_benchmarks("mul_sse\t"),
//     Tests<double, mul_avx::ToTest, chunks>::prepare_benchmarks("mul_avx\t")
// };

void run_benchmarks(Benchmark<double>* benchmark, std::complex<double> *arr, std::size_t to_sum, const std::complex<double> ref) {
    std::cout << benchmark->label;
    for (auto& test_function : benchmark->kernel_tests[0]->tests)
        std::cout << "\t\t" << test_function.param;
    std::cout << std::endl;
    
    Timer t;
    for (auto kernel_test : benchmark->kernel_tests) {
        std::cout << kernel_test->label << "\t";
        
        for (auto& test_function : kernel_test->tests) {
            t.reset();
            std::complex<double> result = test_function.tf(arr, to_sum);
            size_t elapsed = t.elapsed();
                
            printf("\t%lu (%d)", elapsed, abs(result - ref) < 1e-6);
        }
        std::cout << std::endl;
    }
}

int main(int argc, char **argv) {
    if (argc < 2)
        return 1;
    
    std::size_t to_sum = atoi(argv[1]);

    std::complex<double> *arr_to_sum = new std::complex<double>[to_sum];
    for (size_t i = 0; i < to_sum; i++) {
        arr_to_sum[i] = i;
    }
    std::complex<double> sum = sum_simple::sum<double, 1>(arr_to_sum, to_sum);

    std::cout << "Summation: " << av::inst_set << " instruction set" << std::endl;
    run_benchmarks(array_sum_benchmark, arr_to_sum, to_sum, sum);

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
