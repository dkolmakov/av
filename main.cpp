// #pragma GCC optimize("O3", "unroll-loops", "omit-frame-pointer", "inline")
// #pragma GCC option("arch=native", "tune=native", "no-zero-upper")

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <chrono>

#include "common.hpp"
#include "sum_unroll.hpp"
#include "sum_chunked.hpp"
#include "sum_manual.hpp"
#include "mul_unroll.hpp"
#include "mul_manual.hpp"
#include "mul_advanced.hpp"

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

int main(int argc, char **argv) {
    if (argc < 2)
        return 1;
    
    std::size_t to_sum = atoi(argv[1]);
    std::complex<double> *arr = new std::complex<double>[to_sum];
    
    for (size_t i = 0; i < to_sum; i++)
        arr[i] = 1.000001;
        
    Timer t;
    std::complex<double> result = av_unroll::sum(arr, to_sum);
    double elapsed = t.elapsed();
    std::cout << av::inst_set << " unrolled result " << result << " got in " << elapsed << " usec" << std::endl;

    t.reset();
    result = av_chunked::sum(arr, to_sum);
    elapsed = t.elapsed();
    std::cout << av::inst_set << " chunked result " << result << " got in " << elapsed << " usec" << std::endl;
    
    t.reset();
    result = av_manual::sum(arr, to_sum);
    elapsed = t.elapsed();
    std::cout << av::inst_set << " manual result " << result << " got in " << elapsed << " usec" << std::endl;

    t.reset();
    result = av_mul_unroll::mul(arr, to_sum);
    elapsed = t.elapsed();
    std::cout << av::inst_set << " unrolled multiplication result " << result << " got in " << elapsed << " usec" << std::endl;

    t.reset();
    result = av_mul_manual::mul(arr, to_sum);
    elapsed = t.elapsed();
    std::cout << av::inst_set << " manual multiplication result " << result << " got in " << elapsed << " usec" << std::endl;

    t.reset();
    result = av_mul_advanced::mul(arr, to_sum);
    elapsed = t.elapsed();
    std::cout << av::inst_set << " advanced multiplication result " << result << " got in " << elapsed << " usec" << std::endl;
    
    return 0;
}
