#include <stdio.h>
#include <cstdlib>
#include "complex_sum.hpp"

int main(int argc, char **argv) {
    if (argc < 2)
        return 1;
    
    std::size_t to_sum = atoi(argv[1]);
    std::complex<double> *arr = new std::complex<double>[to_sum];
    
    for (size_t i = 0; i < to_sum; i++) {
        arr[i] = i + 1;
    }
    
    asm volatile ("nop;nop;nop;");
    std::complex<double> result = av::sum(arr, to_sum);
    asm volatile ("nop;nop;nop;");
    
    std::cout << "Result: " << result << std::endl;
    return 0;
}
