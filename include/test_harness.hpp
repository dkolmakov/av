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



template<class input_data>
struct TestFunc {
    bool (*tf)(input_data& input);
    std::size_t param;
};


template<class input_data>
struct KernelTest {
    std::size_t size;
    std::string label;
    std::vector<TestFunc<input_data>> tests;

    KernelTest(std::size_t _size, std::string _label) : size(_size), label(_label), tests(size) {}
};


template<class input_data>
struct Benchmark {
    std::size_t size;
    std::string label;

    std::vector<KernelTest<input_data>*> kernel_tests;

    Benchmark(std::size_t _size, std::string _label) : size(_size), label(_label), kernel_tests(size) {}
};


template<class test_func,
         class kernel, class params, std::size_t index>
struct GenTests {
    static void gen(KernelTest<typename test_func::input_data> *kernel_test) {
        kernel_test->tests[index].tf = test_func::template core<params::val, kernel::template core>::compute;
        kernel_test->tests[index].param = params::val;
        GenTests<test_func, kernel, typename params::next, index - 1>::gen(kernel_test);
    }
};

template<class test_func,
         class kernel, class params>
struct GenTests<test_func, kernel, params, 0> {
    static void gen(KernelTest<typename test_func::input_data> *kernel_test) {
        kernel_test->tests[0].tf = test_func::template core<params::val, kernel::template core>::compute;
        kernel_test->tests[0].param = params::val;
    }
};


template<class test_func,
         class kernel, class params, std::size_t index>
struct GenKernelTests {
    typedef typename test_func::input_data input_data;
    
    static void gen(Benchmark<input_data>* benchmark) {
        KernelTest<input_data> *kernel_test = new KernelTest<input_data>(params::total, kernel::val::get_label());
        GenTests<test_func, typename kernel::val, typename params::next, params::total - 1>::gen(kernel_test);

        benchmark->kernel_tests[index] = kernel_test;

        GenKernelTests<test_func, typename kernel::next, params, index - 1>::gen(benchmark);
    }
};

template<class test_func,
         class kernel, class params>
struct GenKernelTests<test_func, kernel, params, 0> {
    typedef typename test_func::input_data input_data;
    
    static void gen(Benchmark<input_data>* benchmark) {
        KernelTest<input_data> *kernel_test = new KernelTest<input_data>(params::total, kernel::val::get_label());
        GenTests<test_func, typename kernel::val, typename params::next, params::total - 1>::gen(kernel_test);

        benchmark->kernel_tests[0] = kernel_test;
    }
};


template<class test_func, class kernels, class params>
struct TestHarness {
    typedef typename test_func::input_data input_data;
    
    static Benchmark<input_data>* prepare_benchmark(std::string label) {
        Benchmark<input_data>* bench = new Benchmark<input_data>(kernels::total, label);

        GenKernelTests<test_func, typename kernels::next, params, kernels::total - 1>::gen(bench);

        return bench;
    }
    
    static void run_benchmark(Benchmark<input_data>* benchmark, std::size_t count) {
        typename test_func::input_data input(count);

        std::cout << benchmark->label;
        for (auto& test_function : benchmark->kernel_tests[0]->tests)
            std::cout << "\t\t" << test_function.param;
        std::cout << std::endl;
        
        Timer t;
        for (auto kernel_test : benchmark->kernel_tests) {
            std::cout << kernel_test->label << "\t";
            
            for (auto& test_function : kernel_test->tests) {
                t.reset();
                bool result = test_function.tf(input);
                size_t elapsed = t.elapsed();
                    
                printf("\t%lu (%d)", elapsed, result);
            }
            std::cout << std::endl;
        }
    }
    
};


