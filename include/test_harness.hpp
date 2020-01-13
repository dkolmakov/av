#pragma once

#include <chrono>
#include <vector>
#include <algorithm>

#include "common.hpp"

namespace av_prof {

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

    static std::string get_label() {
        return std::to_string(val);
    }
};

template<std::size_t index, std::size_t param>
struct CountedParams<index, param> {
    static const std::size_t val = param;
    static const std::size_t total = index + 1;

    static std::string get_label() {
        return std::to_string(val);
    }
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

    static std::string get_label() {
        return val::get_label();
    }
};

template<std::size_t index, class kernel>
struct CountedKernels<index, kernel> {
    typedef kernel val;
    static const std::size_t total = index + 1;

    static std::string get_label() {
        return val::get_label();
    }
};

template<class ... kernels>
struct Kernels {
    typedef CountedKernels<0, kernels...> next;
    static const std::size_t total = next::total;
};


template<class tuple, std::size_t index>
struct ByIndexInt;

template<class tuple>
struct ByIndexInt<tuple, 0> {
    typedef typename tuple::left elem;
};

template<class tuple, std::size_t index>
struct ByIndexInt {
    typedef typename ByIndexInt<typename tuple::right, index - 1>::elem elem;
};


template<class kernel, class param, class kernels, class params, std::size_t kcounter, std::size_t pcounter>
struct CountedPairs;

template<class kernel, class param, class kernels, class params>
struct CountedPairs<kernel, param, kernels, params, 0, 0> {
    typedef CountedPairs<kernel, param, kernels, params, 0, 0> current;
    typedef kernel left;
    typedef param right;
    
    template<std::size_t index>
    struct ByIndex {
        typedef typename ByIndexInt<current, index>::elem elem;
    };

    static std::string get_label() {
        return left::get_label() + " with " + right::get_label();
    }
};

template<class kernel, class param, class kernels, class params, std::size_t kcounter>
struct CountedPairs<kernel, param, kernels, params, kcounter, 0> {
    typedef CountedPairs<kernel, param, kernels, params, kcounter, 0> current;
    typedef CountedPairs<typename kernel::next, typename params::next, kernels, params, kcounter - 1, params::total - 1> next;
    typedef kernel left;
    typedef param right;

    template<std::size_t index>
    struct ByIndex {
        typedef typename ByIndexInt<current, index>::elem elem;
    };

    static std::string get_label() {
        return left::get_label() + " with " + right::get_label();
    }
};

template<class kernel, class param, class kernels, class params, std::size_t kcounter, std::size_t pcounter>
struct CountedPairs {
    typedef CountedPairs<kernel, param, kernels, params, kcounter, pcounter> current;
    typedef CountedPairs<kernel, typename param::next, kernels, params, kcounter, pcounter - 1> next;
    typedef kernel left;
    typedef param right;

    template<std::size_t index>
    struct ByIndex {
        typedef typename ByIndexInt<current, index>::elem elem;
    };

    static std::string get_label() {
        return left::get_label() + " with " + right::get_label();
    }
};

template<class kernels, class params>
struct Pairs {
    typedef CountedPairs<typename kernels::next, typename params::next, kernels, params, kernels::total - 1, params::total - 1> next;
    static const std::size_t total = kernels::total * params::total;
};

typedef Kernels<std::nullptr_t> ListTerminator;

template<std::size_t index, class elem, class ... elems>
struct CombinationsInt {
    typedef Pairs<elem, typename CombinationsInt<index + 1, elems...>::val> val;
    static const std::size_t total = val::total;
};

template<std::size_t index, class elem>
struct CombinationsInt<index, elem> {
    typedef Pairs<elem, ListTerminator> val;
    static const std::size_t total = index + 1;
};


template<class ... elems>
struct Combinations {
    typedef typename CombinationsInt<0, elems...>::val val;
    static const std::size_t total = val::total;
};



template<class pairs, std::size_t index>
struct PairsPrinter {
    static void print() {
        std::cout << pairs::get_label() << std::endl;
        PairsPrinter<typename pairs::next, index - 1>::print();
    }
};

template<class pairs>
struct PairsPrinter<pairs, 0> {
    static void print() {
        std::cout << pairs::get_label() << std::endl;
    }
};



template<class input_data>
struct TestFunc {
    bool (*tf)(input_data& input);
    std::string (*get_label)(void);
    size_t elapsed = 0;
    bool result = true;
};


template<class input_data>
struct Benchmark {
    std::string label;
    std::vector<TestFunc<input_data>> tests;

    Benchmark(std::size_t _size, std::string _label) : label(_label), tests(_size) {}

    void run(std::size_t count) {
        input_data input(count);

        Timer t;
        for (auto& test : tests) {
            t.reset();
            test.result = test.result && test.tf(input);
            test.elapsed += t.elapsed();
        }
    }
    
    void print_results() {
        std::sort(tests.begin(), tests.end(), [](TestFunc<input_data> i, TestFunc<input_data> j) { return (i.elapsed < j.elapsed);});
        
        for (auto& test : tests) {
            std::cout << test.get_label() << "\t";
            printf("\t%lu (%d)", test.elapsed, test.result);
            std::cout << std::endl;
        }
    }            
};


template<class test_func, class param_tuple, std::size_t index>
struct GenTests {
    typedef typename test_func::input_data input_data;
    
    static void gen(std::vector<TestFunc<input_data>>& tests) {
        tests[index].tf = test_func::template core<param_tuple>::compute;
        tests[index].get_label = test_func::template core<param_tuple>::get_label;
        GenTests<test_func, typename param_tuple::next, index - 1>::gen(tests);
    }
};

template<class test_func, class param_tuple>
struct GenTests<test_func, param_tuple, 0> {
    typedef typename test_func::input_data input_data;
    
    static void gen(std::vector<TestFunc<input_data>>& tests) {
        tests[0].tf = test_func::template core<param_tuple>::compute;
        tests[0].get_label = test_func::template core<param_tuple>::get_label;
    }
};


template<class test_func, class param_tuples>
struct TestHarness {
    typedef typename test_func::input_data input_data;
    
    static Benchmark<input_data>* prepare_benchmark(std::string label) {
        Benchmark<input_data>* bench = new Benchmark<input_data>(param_tuples::total, label);

        GenTests<test_func, typename param_tuples::next, param_tuples::total - 1>::gen(bench->tests);

        return bench;
    }
    
};


}
