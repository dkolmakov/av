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

template<std::size_t index,
         template<typename, std::size_t, template<typename, std::size_t> class chunk_sum> class test_func,   
         class test_kernel, 
         class ... test_kernels>
struct CountedKernels {
    typedef CountedKernels<index + 1, test_func, test_kernels...> next;

    static std::string get_label() {
        return test_kernel::get_label();
    }
    
    template<class T, std::size_t sz>
    struct val {
        static std::complex<T> compute(std::complex<T>* data, std::size_t size) {
            return test_func<T, sz, test_kernel::template core>::compute(data, size);
        }
    };
    
    static const std::size_t total = next::total;
};

template<std::size_t index,
         template<typename, std::size_t, template<typename, std::size_t> class chunk_sum> class test_func,
         class test_kernel>
struct CountedKernels<index, test_func, test_kernel> {

    static std::string get_label() {
        return test_kernel::get_label();
    }
    
    template<class T, std::size_t sz>
    struct val{
        static std::complex<T> compute(std::complex<T>* data, std::size_t size) {
            return test_func<T, sz, test_kernel::template core>::compute(data, size);
        }
    };
    
    static const std::size_t total = index + 1;
};

template<template<typename, std::size_t, template<typename, std::size_t> class chunk_sum> class test_func, 
         class ... test_kernels>
struct Kernels {
    typedef CountedKernels<0, test_func, test_kernels...> next; 
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
        wrapper->benchmarks[index].tf = F<T, test_vals::val>::compute;
        wrapper->benchmarks[index].param = test_vals::val;
        GenBenchmark<T, F, typename test_vals::next, index - 1>::gen(wrapper);
    }
};

template<class T, template<class TT, std::size_t size> class F, class test_vals>
struct GenBenchmark<T, F, test_vals, 0> {
    static void gen(BenchmarkWrapper<T> *wrapper) {
        wrapper->benchmarks[0].tf = F<T, test_vals::val>::compute;
        wrapper->benchmarks[0].param = test_vals::val;
    }
};

template<class T, class test_kernel, class test_vals, std::size_t index>
struct GenWrapper;

template<class T, class test_kernel, class test_vals>
struct GenWrapper<T, test_kernel, test_vals, 0> {
    static void gen(std::vector<BenchmarkWrapper<T>*>* wrappers) {
        BenchmarkWrapper<T> *wrapper = new BenchmarkWrapper<T>(test_vals::total, test_kernel::get_label());
        GenBenchmark<T, test_kernel::template val, typename test_vals::next, test_vals::total - 1>::gen(wrapper);
        wrappers->push_back(wrapper);
    }
};

template<class T, class test_kernel, class test_vals, std::size_t index>
struct GenWrapper {
    static void gen(std::vector<BenchmarkWrapper<T>*>* wrappers) {
        BenchmarkWrapper<T> *wrapper = new BenchmarkWrapper<T>(test_vals::total, test_kernel::get_label());
        GenBenchmark<T, test_kernel::template val, typename test_vals::next, test_vals::total - 1>::gen(wrapper);
        wrappers->push_back(wrapper);
        
        GenWrapper<T, typename test_kernel::next, test_vals, index - 1>::gen(wrappers);
    }
};


template<class T, class test_kernels, class test_vals>
struct Tests {
    static std::vector<BenchmarkWrapper<T>*>* prepare_benchmarks() {
        std::vector<BenchmarkWrapper<T>*>* wrappers = new std::vector<BenchmarkWrapper<T>*>(); 

        GenWrapper<T, typename test_kernels::next, test_vals, test_kernels::total - 1>::gen(wrappers);
        
        return wrappers;
    }
};


