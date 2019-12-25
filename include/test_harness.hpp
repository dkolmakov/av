#pragma once

#include <chrono>
#include <vector>
#include "common.hpp"

class Timer
{
public:
    Timer() : start(clock_t::now()) {}
    void reset() { start = clock_t::now(); }
    std::size_t elapsed() const {
        return std::chrono::duration_cast<std::chrono::microseconds>
            (clock_t::now() - start).count(); }

private:
    typedef std::chrono::high_resolution_clock clock_t;
    std::chrono::time_point<clock_t> start;
};



template<std::size_t index, std::size_t param, std::size_t... params_left>
struct CountedParams {
    typedef CountedParams<index + 1, params_left...> next; 
    static const std::size_t val = param;
    static const std::size_t total = next::total;
};

template<std::size_t index, std::size_t param>
struct CountedParams<index, param> {
    static const std::size_t val = param;
    static const std::size_t total = index + 1;
};

template<std::size_t... params>
struct KernelParameters {
    typedef CountedParams<0, params...> next; 
    static const std::size_t total = next::total;
};



template<std::size_t index,
         template<typename, std::size_t, template<typename, std::size_t> class chunk_sum> class test_func,   
         class kernel, 
         class ... kernels_left>
struct CountedKernels {
    typedef CountedKernels<index + 1, test_func, kernels_left...> next;

    static std::string get_label() {
        return kernel::get_label();
    }
    
    template<class T, std::size_t sz>
    struct val {
        static std::complex<T> compute(std::complex<T>* data, std::size_t size) {
            return test_func<T, sz, kernel::template core>::compute(data, size);
        }
    };
    
    static const std::size_t total = next::total;
};

template<std::size_t index,
         template<typename, std::size_t, template<typename, std::size_t> class chunk_sum> class test_func,
         class kernel>
struct CountedKernels<index, test_func, kernel> {

    static std::string get_label() {
        return kernel::get_label();
    }
    
    template<class T, std::size_t sz>
    struct val{
        static std::complex<T> compute(std::complex<T>* data, std::size_t size) {
            return test_func<T, sz, kernel::template core>::compute(data, size);
        }
    };
    
    static const std::size_t total = index + 1;
};

template<template<typename, std::size_t, template<typename, std::size_t> class chunk_sum> class test_func, 
         class ... kernels>
struct Kernels {
    typedef CountedKernels<0, test_func, kernels...> next; 
    static const std::size_t total = next::total;
};



template<class T>
struct TestFunc {
    std::complex<T> (*tf)(std::complex<T>*, std::size_t);
    std::size_t param;
};


template<class T>
struct KernelTest {
  std::size_t size;
  std::string label;
  std::vector<TestFunc<T>> tests;
  
  KernelTest(std::size_t _size, std::string _label) : size(_size), label(_label), tests(size) {}
};

template<class T>
struct Benchmark {
  std::size_t size;
  std::string label;
  
  std::vector<KernelTest<T>*> kernel_tests;
  
  Benchmark(std::size_t _size, std::string _label) : size(_size), label(_label), kernel_tests(size) {
  }
  
};


template<class T, template<class TT, std::size_t size> class F, class test_vals, std::size_t index>
struct GenTests {
    static void gen(KernelTest<T> *kernel_test) {
        kernel_test->tests[index].tf = F<T, test_vals::val>::compute;
        kernel_test->tests[index].param = test_vals::val;
        GenTests<T, F, typename test_vals::next, index - 1>::gen(kernel_test);
    }
};

template<class T, template<class TT, std::size_t size> class F, class test_vals>
struct GenTests<T, F, test_vals, 0> {
    static void gen(KernelTest<T> *kernel_test) {
        kernel_test->tests[0].tf = F<T, test_vals::val>::compute;
        kernel_test->tests[0].param = test_vals::val;
    }
};


template<class T, class test_kernel, class test_vals, std::size_t index>
struct GenKernelTests {
    static void gen(Benchmark<T>* benchmark) {
        KernelTest<T> *kernel_test = new KernelTest<T>(test_vals::total, test_kernel::get_label());
        GenTests<T, test_kernel::template val, typename test_vals::next, test_vals::total - 1>::gen(kernel_test);
        
        benchmark->kernel_tests[index] = kernel_test;
        
        GenKernelTests<T, typename test_kernel::next, test_vals, index - 1>::gen(benchmark);
    }
};

template<class T, class test_kernel, class test_vals>
struct GenKernelTests<T, test_kernel, test_vals, 0> {
    static void gen(Benchmark<T>* benchmark) {
        KernelTest<T> *kernel_test = new KernelTest<T>(test_vals::total, test_kernel::get_label());
        GenTests<T, test_kernel::template val, typename test_vals::next, test_vals::total - 1>::gen(kernel_test);
        
        benchmark->kernel_tests[0] = kernel_test;
    }
};


template<class T, class test_kernels, class test_vals>
struct TestHarness {
    static Benchmark<T>* prepare_benchmark(std::string label) {
        Benchmark<T>* bench = new Benchmark<T>(test_kernels::total, label); 

        GenKernelTests<T, typename test_kernels::next, test_vals, test_kernels::total - 1>::gen(bench);
        
        return bench;
    }
};


