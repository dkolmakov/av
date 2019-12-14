#pragma once

#include <array>
#include "common.hpp"

template<class T>
struct Benchmark {
    std::complex<T> (*tf)(std::complex<T>*, std::size_t);
};

template<class T, template<class TT, std::size_t size> class F, std::size_t N, std::size_t test_vals[N], std::size_t index>
struct GenBenchmark;

template<class T, template<class TT, std::size_t size> class F, std::size_t N, std::size_t test_vals[N]>
struct GenBenchmark<T, F, N, test_vals, 0> {
    static void gen(Benchmark<T> *benchmarks) {
        benchmarks[0].tf = F<T, test_vals[0]>::to_test;
    }
};

template<class T, template<class TT, std::size_t size> class F, std::size_t N, std::size_t test_vals[N], std::size_t index>
struct GenBenchmark {
    static void gen(Benchmark<T> *benchmarks) {
        GenBenchmark<T, F, N, test_vals, index - 1>::gen(benchmarks);
        benchmarks[index].tf = F<T, test_vals[index]>::to_test;
    }
};


template<class T, template<class TT, std::size_t size> class F, std::size_t N, std::size_t test_vals[N]>
struct Tests {
    static Benchmark<T> *prepare_benchmarks() {
        Benchmark<T> *benchmarks = new Benchmark<T>[N];
        GenBenchmark<T, F, N, test_vals, N - 1>::gen(benchmarks);
    }
};
