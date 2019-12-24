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

template<std::size_t index, std::size_t test_val, std::size_t... test_vals>
struct CountedChunkSizes {
    typedef CountedChunkSizes<index + 1, test_vals...> next; 
    static const std::size_t val = test_val;
    static const std::size_t total = next::total;
    static const bool last = false;
};

template<std::size_t index, std::size_t test_val>
struct CountedChunkSizes<index, test_val> {
    static const std::size_t val = test_val;
    static const std::size_t total = index + 1;
    static const bool last = true;
};


template<std::size_t... test_vals>
struct ChunkSizes {
    typedef CountedChunkSizes<0, test_vals...> next; 
    static const std::size_t total = next::total;
};

template<std::size_t index, template<class TT, std::size_t size> class test_kernel, template<class TT, std::size_t size> class ... test_kernels>
struct CountedKernels {
    typedef CountedKernels<index + 1, test_kernels...> next;

    template<class T, std::size_t sz>
    static std::complex<T> func(std::complex<T>* data, std::size_t size) {
        return test_kernel<T, sz>::compute(data, size);
    }

    static const std::size_t total = next::total;
    static const bool last = false;
};

template<std::size_t index, template<class TT, std::size_t sz> class test_kernel>
struct CountedKernels<index, test_kernel> {
    template<class T, std::size_t sz>
    static std::complex<T> func(std::complex<T>* data, std::size_t size) {
        return test_kernel<T, sz>::compute(data, size);
    }
    
    static const std::size_t total = index + 1;
    static const bool last = true;
};


template<template<class TT, std::size_t sz> class ... test_kernels>
struct Kernels {
    typedef CountedKernels<0, test_kernels...> next; 
    static const std::size_t total = next::total;
};


template<class T>
struct Benchmark {
    std::complex<T> (*tf)(std::complex<T>*, std::size_t);
    std::size_t param;
};

template<class T>
struct BenchmarkWrapper {
  std::size_t size;
  std::string label;
  std::vector<Benchmark<T>> benchmarks;
  
  BenchmarkWrapper(std::size_t _size, std::string _label) : size(_size), label(_label), benchmarks(size) {
  }
  
  ~BenchmarkWrapper() {
      delete[] benchmarks;
  }
  
};

template<class T, template<class TT, std::size_t size> class F, class test_vals, std::size_t index>
struct GenBenchmark {
    static void gen(BenchmarkWrapper<T> *wrapper) {
        wrapper->benchmarks[index].tf = F<T, test_vals::val>::to_test;
        wrapper->benchmarks[index].param = test_vals::val;
        GenBenchmark<T, F, typename test_vals::next, index - 1>::gen(wrapper);
    }
};

template<class T, template<class TT, std::size_t size> class F, class test_vals>
struct GenBenchmark<T, F, test_vals, 0> {
    static void gen(BenchmarkWrapper<T> *wrapper) {
        wrapper->benchmarks[0].tf = F<T, test_vals::val>::to_test;
        wrapper->benchmarks[0].param = test_vals::val;
    }
};


template<class T, template<class TT, std::size_t size> class F, class test_vals>
struct Tests {
    static BenchmarkWrapper<T> *prepare_benchmarks(std::string label) {
        BenchmarkWrapper<T> *wrapper = new BenchmarkWrapper<T>(test_vals::total, label);
        
        GenBenchmark<T, F, typename test_vals::next, test_vals::total - 1>::gen(wrapper);
        
        return wrapper;
    }
};


