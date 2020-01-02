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


template<class input_data, class param, class ... params>
struct KernelTest {
    const bool is_last = false;
    typedef KernelTest<input_data, params...> Tests;
    
    std::string label;
    std::vector<Tests*> tests;

    KernelTest(std::size_t _size, std::string _label) : label(_label), tests(_size) {}
};

template<class input_data, class param>
struct KernelTest<input_data, param> {
    const bool is_last = true;
    std::string label;
    std::vector<TestFunc<input_data>> tests;

    KernelTest(std::size_t _size, std::string _label) : label(_label), tests(_size) {}

};


template<class input_data, class ... params>
struct Benchmark {
    std::string label;
    typedef KernelTest<input_data, params...> Tests;
    
    std::vector<Tests*> kernel_tests;

    Benchmark(std::size_t _size, std::string _label) : label(_label), kernel_tests(_size) {}

    void run(std::size_t count) {
        input_data input(count);

        std::cout << label;
        for (auto& test_function : kernel_tests[0]->tests)
            std::cout << "\t\t" << test_function.param;
        std::cout << std::endl;
        
        Timer t;
        for (auto kernel_test : kernel_tests) {
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


template<class test_func, class params, std::size_t index>
struct GenTests {
    typedef typename test_func::input_data input_data;
    
    static void gen(std::vector<TestFunc<input_data>>& tests) {
        tests[index].tf = test_func::template core<params::val>::compute;
        tests[index].param = params::val;
        GenTests<test_func, typename params::next, index - 1>::gen(tests);
    }
};

template<class test_func, class params>
struct GenTests<test_func, params, 0> {
    typedef typename test_func::input_data input_data;
    
    static void gen(std::vector<TestFunc<input_data>>& tests) {
        tests[0].tf = test_func::template core<params::val>::compute;
        tests[0].param = params::val;
    }
};

template<class test_func, class kernel, std::size_t index, class ... params>
struct IterOverCurrentLevel;

template<class test_func, class param, class ... params>
struct GenKernelTests {
    typedef typename test_func::input_data input_data;
    
    static void gen(KernelTest<input_data, params...>** kernel_test) {
        KernelTest<input_data, params...> *to_add = new KernelTest<input_data, params...>(param::total, param::val::get_label());
        *kernel_test = to_add;

        IterOverCurrentLevel<test_func, typename param::next, param::total - 1, params...>::gen(to_add->tests);
    }
};

template<class test_func, class param>
struct GenKernelTests<test_func, param> {
    typedef typename test_func::input_data input_data;
    
    static void gen(KernelTest<input_data, param>** kernel_test) {
        KernelTest<input_data, param> *to_add = new KernelTest<input_data, param>(param::total, param::val::get_label());
        *kernel_test = to_add;
        
        GenTests<typename test_func::template core<param>, typename param::next, param::total - 1>::gen(to_add->tests);
    }
};


template<class test_func, class kernel, std::size_t index, class ... params>
struct IterOverCurrentLevel {
    typedef typename test_func::input_data input_data;
    
    static void gen(std::vector<KernelTest<input_data, params...>*>& kernel_tests) {
        GenKernelTests<typename test_func::template core<kernel>, params...>::gen(&kernel_tests[index]);
        IterOverCurrentLevel<test_func, typename kernel::next, index - 1, params...>::gen(kernel_tests);
    }
};

template<class test_func, class kernel, class ... params>
struct IterOverCurrentLevel<test_func, kernel, 0, params...> {
    typedef typename test_func::input_data input_data;
    
    static void gen(std::vector<KernelTest<input_data, params...>*>& kernel_tests) {
        GenKernelTests<typename test_func::template core<kernel>, params...>::gen(&kernel_tests[0]);
    }
};


template<class test_func, class param, class ... params>
struct TestHarness {
    typedef typename test_func::input_data input_data;
    
    static Benchmark<input_data, param, params...>* prepare_benchmark(std::string label) {
        Benchmark<input_data, param, params...>* bench = new Benchmark<input_data, param, params...>(param::total, label);

        IterOverCurrentLevel<test_func, typename param::next, param::total - 1, params...>::gen(bench->kernel_tests);

        return bench;
    }
    
};


