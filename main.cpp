// #pragma GCC optimize("O3", "unroll-loops", "omit-frame-pointer", "inline")
// #pragma GCC option("arch=native", "tune=native", "no-zero-upper")

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <chrono>

#include "common.hpp"
#include "sum_simple.hpp"
#include "sum_unroll.hpp"
#include "sum_chunked.hpp"
#include "sum_manual.hpp"
#include "mul_simple.hpp"
#include "mul_unroll.hpp"
#include "mul_manual.hpp"
#include "mul_avx.hpp"

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<std::chrono::microseconds>
            (clock_::now() - beg_).count(); }

private:
    typedef std::chrono::high_resolution_clock clock_;
    std::chrono::time_point<clock_> beg_;
};

const struct av::CalculationTask<double> tasks[] = {
    {"Simple summation", av_simple::sum},
    {"Unrolled summation chunk=2", av_unroll::sum<double, 1>},
    {"Unrolled summation chunk=2", av_unroll::sum<double, 2>},
    {"Unrolled summation chunk=4", av_unroll::sum<double, 4>},
    {"Unrolled summation chunk=8", av_unroll::sum<double, 8>},
    {"Unrolled summation chunk=16", av_unroll::sum<double, 16>},
    {"Unrolled summation chunk=32", av_unroll::sum<double, 32>},
    {"Chunked summation chunk=1", av_chunked::sum<double, 1>},
    {"Chunked summation chunk=2", av_chunked::sum<double, 2>},
    {"Chunked summation chunk=4", av_chunked::sum<double, 4>},
    {"Chunked summation chunk=8", av_chunked::sum<double, 8>},
    {"Manual summation", av_manual::sum},
    {"Simple multiplication", av_mul_simple::mul},
    {"Unrolled multiplication", av_mul_unroll::mul},
    {"Manual multiplication", av_mul_manual::mul},
    {"Advanced multiplication 4", av_mul_avx::mul<double, 4>},
    {"Advanced multiplication 8", av_mul_avx::mul<double, 8>},
    {"Advanced multiplication 16", av_mul_avx::mul<double, 16>},
    {"Advanced multiplication 32", av_mul_avx::mul<double, 32>},
    {"Advanced multiplication 64", av_mul_avx::mul<double, 64>}
};

int main(int argc, char **argv) {
    if (argc < 2)
        return 1;
    
    std::size_t to_sum = atoi(argv[1]);
    std::complex<double> *arr = new std::complex<double>[to_sum];
    
    for (size_t i = 0; i < to_sum; i++)
        arr[i] = 1.000001;
        
    std::cout << "Starting tests for: " << av::inst_set << " instruction set" << std::endl;

    Timer t;
    for (auto& task : tasks) {
        t.reset();
        std::complex<double> result = task.func(arr, to_sum);
        double elapsed = t.elapsed();
        std::cout << task.label << " result " << result << " got in " << elapsed << " usec" << std::endl;
    }
    
    return 0;
}
