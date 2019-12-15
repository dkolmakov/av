#pragma once

#include <chrono>
#include "common.hpp"

class Timer
{
public:
    Timer() : start(clock_t::now()) {}
    void reset() { start = clock_t::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<std::chrono::microseconds>
            (clock_t::now() - start).count(); }

private:
    typedef std::chrono::high_resolution_clock clock_t;
    std::chrono::time_point<clock_t> start;
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
  Benchmark<T> *benchmarks;
  
  BenchmarkWrapper(std::size_t _size, std::string _label) : size(_size), label(_label) {
      benchmarks = new Benchmark<T>[size];
  }
  
  ~BenchmarkWrapper() {
      delete[] benchmarks;
  }
  
};

template<class T, template<class TT, std::size_t size> class F, std::size_t index, std::size_t test_val, std::size_t... test_vals>
struct GenBenchmark {
    static void gen(BenchmarkWrapper<T> *wrapper) {
        GenBenchmark<T, F, index - 1, test_vals...>::gen(wrapper);
        wrapper->benchmarks[index].tf = F<T, test_val>::to_test;
        wrapper->benchmarks[index].param = test_val;
    }
};

template<class T, template<class TT, std::size_t size> class F, std::size_t index, std::size_t test_val>
struct GenBenchmark<T, F, index, test_val> {
    static void gen(BenchmarkWrapper<T> *wrapper) {
        wrapper->benchmarks[0].tf = F<T, test_val>::to_test;
        wrapper->benchmarks[0].param = test_val;
    }
};


template<class T, template<class TT, std::size_t size> class F, std::size_t N, std::size_t... test_vals>
struct Tests {
    static BenchmarkWrapper<T> *prepare_benchmarks(std::string label) {
        BenchmarkWrapper<T> *wrapper = new BenchmarkWrapper<T>(N, label);
        
        GenBenchmark<T, F, N - 1, test_vals...>::gen(wrapper);
        
        return wrapper;
    }
};


