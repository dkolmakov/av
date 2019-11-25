// #pragma GCC optimize("O3", "unroll-loops", "omit-frame-pointer", "inline")
// #pragma GCC option("arch=native", "tune=native", "no-zero-upper")

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <chrono>

#include "common.hpp"
#include "complex_sum.hpp"

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
    
    for (size_t i = 0; i < to_sum; i++) {
        arr[i] = i + 1;
    }
    
    std::cout << "Runner " << av::inst_set << std::endl;
    
    Timer t;
//     asm volatile ("nop;nop;nop;");
    std::complex<double> result = av::sum(arr, to_sum);
//     asm volatile ("nop;nop;nop;");
    double elapsed = t.elapsed();
    
    std::cout << "Result " << result << " elapsed in " << elapsed << " usec" << std::endl;

    return 0;
}
