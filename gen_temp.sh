#!/usr/bin/env bash

echo """
#include <iostream>
#include <cstdlib>

#include \"${1}.hpp\"

int main(int argc, char **argv) {
    if (argc < 2)
        return 1;
    
    std::size_t to_sum = atoi(argv[1]);
    std::complex<double> *arr = new std::complex<double>[to_sum];
    
    for (size_t i = 0; i < to_sum; i++) {
        arr[i] = 1.000001;
    }
        
    std::complex<double> sum = ${1}::ToTest<double, ${2}>::to_test(arr, to_sum);

    std::cout << \"Result: \" << sum << std::endl;
    
    return 0;
}
""" > temp.cpp
