#pragma once

#include <chrono>
#include <vector>
#include "common.hpp"

class Timer
{
public:
    Timer() : start(clock_t::now()) {}
    void reset() {
        start = clock_t::now();
    }
    std::size_t elapsed() const {
        return std::chrono::duration_cast<std::chrono::microseconds>
               (clock_t::now() - start).count();
    }

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



template<std::size_t index, class kernel, class ... kernels_left>
struct CountedKernels {
    typedef CountedKernels<index + 1, kernels_left...> next;
    typedef kernel val;
    static const std::size_t total = next::total;
};

template<std::size_t index, class kernel>
struct CountedKernels<index, kernel> {
    typedef kernel val;
    static const std::size_t total = index + 1;
};

template<class ... kernels>
struct Kernels {
    typedef CountedKernels<0, kernels...> next;
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

    Benchmark(std::size_t _size, std::string _label) : size(_size), label(_label), kernel_tests(size) {}
};


template<class T,
         template<typename, std::size_t, template<typename, std::size_t> class chunk_op> class test_func,
         class kernel, class params, std::size_t index>
struct GenTests {
    static void gen(KernelTest<T> *kernel_test) {
        kernel_test->tests[index].tf = test_func<T, params::val, kernel::template core>::compute;
        kernel_test->tests[index].param = params::val;
        GenTests<T, test_func, kernel, typename params::next, index - 1>::gen(kernel_test);
    }
};

template<class T,
         template<typename, std::size_t, template<typename, std::size_t> class chunk_op> class test_func,
         class kernel, class params>
struct GenTests<T, test_func, kernel, params, 0> {
    static void gen(KernelTest<T> *kernel_test) {
        kernel_test->tests[0].tf = test_func<T, params::val, kernel::template core>::compute;
        kernel_test->tests[0].param = params::val;
    }
};


template<class T,
         template<typename, std::size_t, template<typename, std::size_t> class chunk_op> class test_func,
         class kernel, class params, std::size_t index>
struct GenKernelTests {
    static void gen(Benchmark<T>* benchmark) {
        KernelTest<T> *kernel_test = new KernelTest<T>(params::total, kernel::val::get_label());
        GenTests<T, test_func, typename kernel::val, typename params::next, params::total - 1>::gen(kernel_test);

        benchmark->kernel_tests[index] = kernel_test;

        GenKernelTests<T, test_func, typename kernel::next, params, index - 1>::gen(benchmark);
    }
};

template<class T,
         template<typename, std::size_t, template<typename, std::size_t> class chunk_op> class test_func,
         class kernel, class params>
struct GenKernelTests<T, test_func, kernel, params, 0> {
    static void gen(Benchmark<T>* benchmark) {
        KernelTest<T> *kernel_test = new KernelTest<T>(params::total, kernel::val::get_label());
        GenTests<T, test_func, typename kernel::val, typename params::next, params::total - 1>::gen(kernel_test);

        benchmark->kernel_tests[0] = kernel_test;
    }
};


template<class T,
         template<typename, std::size_t, template<typename, std::size_t> class chunk_op> class test_func,
         class kernels, class params>
struct TestHarness {
    static Benchmark<T>* prepare_benchmark(std::string label) {
        Benchmark<T>* bench = new Benchmark<T>(kernels::total, label);

        GenKernelTests<T, test_func, typename kernels::next, params, kernels::total - 1>::gen(bench);

        return bench;
    }
};


