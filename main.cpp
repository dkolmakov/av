// #pragma GCC optimize("O3", "unroll-loops", "omit-frame-pointer", "inline")
// #pragma GCC option("arch=native", "tune=native", "no-zero-upper")

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <array>

#include "common.hpp"
#include "sum_simple.hpp"
#include "sum_unroll.hpp"
#include "sum_chunked.hpp"
#include "sum_manual.hpp"
#include "mul_simple.hpp"
#include "mul_unroll.hpp"
#include "mul_manual.hpp"
#include "mul_avx.hpp"

#include "test_harness.hpp"

// static constexpr std::size_t chunk_sizes[] = {1, 2, 4, 8, 16, 24, 32, 48, 64};
// constexpr std::size_t chunks_num = sizeof(chunk_sizes) / sizeof(std::size_t);
// constexpr std::array<std::size_t, 9> chunks_array = {1, 2, 4, 8, 16, 24, 32, 48, 64};

#define CHUNKS 1, 2, 4, 8, 16, 24, 32, 48, 64

const struct BenchmarkWrapper<double>* tasks[] = {
    Tests<double, av_simple::ToTest, 1, 1>::prepare_benchmarks("Simple summation"),
    Tests<double, av_unroll::ToTest, 9, CHUNKS>::prepare_benchmarks("Unrolled summation"),
    Tests<double, av_chunked::ToTest, 9, CHUNKS>::prepare_benchmarks("Chunked summation"),
    Tests<double, av_manual::ToTest, 1, 1>::prepare_benchmarks("Manual summation"),
    
    Tests<double, av_mul_simple::ToTest, 1, 1>::prepare_benchmarks("Simple multiplication"),
    Tests<double, av_mul_unroll::ToTest, 9, CHUNKS>::prepare_benchmarks("Unrolled multiplication"),
    Tests<double, av_mul_manual::ToTest, 1, 1>::prepare_benchmarks("Manual multiplication"),
    Tests<double, av_mul_avx::ToTest, 9, CHUNKS>::prepare_benchmarks("Advanced multiplication")
};


int main(int argc, char **argv) {
    if (argc < 2)
        return 1;
    
    std::size_t to_sum = atoi(argv[1]);
    std::complex<double> *arr = new std::complex<double>[to_sum];
    
    for (size_t i = 0; i < to_sum; i++)
        arr[i] = 1.000001;
        
    std::cout << std::endl << std::endl << "Starting tests for: " << av::inst_set << " instruction set" << std::endl;

    Timer t;
    for (auto task : tasks) {
        for (std::size_t i = 0; i < task->size; i++) {
            auto& bench = task->benchmarks[i];
            
            t.reset();
            std::complex<double> result = bench.tf(arr, to_sum);
            double elapsed = t.elapsed();
            
            std::cout << task->label<< " chunk " << bench.param << " result " << result << " got in " << elapsed << " usec" << std::endl;
        }
    }
    
    return 0;
}
