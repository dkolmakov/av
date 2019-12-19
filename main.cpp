// #pragma GCC optimize("O3", "unroll-loops", "omit-frame-pointer", "inline")
// #pragma GCC option("arch=native", "tune=native", "no-zero-upper")

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <array>
#include <vector>

#include "common.hpp"
#include "sum_simple.hpp"
#include "sum_unroll.hpp"
#include "sum_chunked.hpp"
#include "sum_man_sse.hpp"
#include "sum_adv_sse.hpp"

#include "mul_simple.hpp"
#include "mul_unroll.hpp"
#include "mul_manual.hpp"
#include "mul_avx.hpp"
#include "mul_sse.hpp"

#include "test_harness.hpp"

// static constexpr std::size_t chunk_sizes[] = {1, 2, 4, 8, 16, 24, 32, 48, 64};
// constexpr std::size_t chunks_num = sizeof(chunk_sizes) / sizeof(std::size_t);
// constexpr std::array<std::size_t, 9> chunks_array = {1, 2, 4, 8, 16, 24, 32, 48, 64};

#define CHUNKS 1, 2, 4, 8, 16, 24, 32, 48, 64
#define CHUNKS_NUM 9

// #define CHUNKS 16
// #define CHUNKS_NUM 1

std::vector<BenchmarkWrapper<double>*> sum_tasks = {
    Tests<double, sum_simple::ToTest, CHUNKS_NUM, CHUNKS>::prepare_benchmarks("sum_simple"),
    Tests<double, sum_unroll::ToTest, CHUNKS_NUM, CHUNKS>::prepare_benchmarks("sum_unroll"),
    Tests<double, sum_chunked::ToTest, CHUNKS_NUM, CHUNKS>::prepare_benchmarks("sum_chunked"),
    Tests<double, sum_man_sse::ToTest, CHUNKS_NUM, CHUNKS>::prepare_benchmarks("sum_man_sse"),
    Tests<double, sum_adv_sse::ToTest, CHUNKS_NUM, CHUNKS>::prepare_benchmarks("sum_adv_sse"),
};

std::vector<BenchmarkWrapper<double>*> mul_tasks = {
    Tests<double, mul_simple::ToTest, CHUNKS_NUM, CHUNKS>::prepare_benchmarks("mul_simple"),
    Tests<double, mul_unroll::ToTest, CHUNKS_NUM, CHUNKS>::prepare_benchmarks("mul_unroll"),
    Tests<double, mul_manual::ToTest, CHUNKS_NUM, CHUNKS>::prepare_benchmarks("mul_manual"),
    Tests<double, mul_sse::ToTest, CHUNKS_NUM, CHUNKS>::prepare_benchmarks("mul_sse\t"),
    Tests<double, mul_avx::ToTest, CHUNKS_NUM, CHUNKS>::prepare_benchmarks("mul_avx\t")
};

void run_benchmarks(std::vector<BenchmarkWrapper<double>*> tasks, std::complex<double> *arr, std::size_t to_sum, const std::complex<double> ref) {
    std::cout << "\t";
    for (auto& bench : tasks[0]->benchmarks)
        std::cout << "\t\t" << bench.param;
    std::cout << std::endl;
    
    Timer t;
    for (auto task : tasks) {
        std::cout << task->label << "\t";
        for (std::size_t i = 0; i < task->size; i++) {
            auto& bench = task->benchmarks[i];
            
            t.reset();
            std::complex<double> result = bench.tf(arr, to_sum);
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
    std::complex<double> sum = sum_simple::sum(arr_to_sum, to_sum);

    std::cout << "Summation: " << av::inst_set << " instruction set" << std::endl;
    run_benchmarks(sum_tasks, arr_to_sum, to_sum, sum);

    std::complex<double> *arr_to_mul = new std::complex<double>[to_sum];
    for (size_t i = 0; i < to_sum; i++) {
        arr_to_mul[i] = 1;
        if ((i % (size_t)(0.1 * to_sum)) == 0) {
            arr_to_mul[i] = 1 + i / (size_t)(0.1 * to_sum);
        }
    }
    std::complex<double> mul = mul_simple::mul(arr_to_mul, to_sum);

    std::cout << "Multiplication: " << av::inst_set << " instruction set" << std::endl;
    run_benchmarks(mul_tasks, arr_to_mul, to_sum, mul);
   
    return 0;
}
