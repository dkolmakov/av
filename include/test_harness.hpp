#pragma once

#include <chrono>
#include <vector>
#include <algorithm>

#include "combinations.hpp"
#include "progress_bar.hpp"

namespace av {
namespace impl {
    
template<class input_data>
struct TestFunc {
    bool (*tf)(input_data& input);
    std::string (*get_label)(void);
    size_t elapsed = 0;
    bool result = true;
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

}

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


template<class input_data>
struct Benchmark {
    std::string label;
    std::vector<impl::TestFunc<input_data>> tests;

    Benchmark(std::size_t _size, std::string _label) : label(_label), tests(_size) {}

    void run(std::size_t count) {
        input_data input(count);
        std::size_t current = 0;
        ProgressBar<size_t> bar(tests.size(), current);

        Timer t;
        for (auto& test : tests) {
            t.reset();
            test.result = test.result && test.tf(input);
            test.elapsed += t.elapsed();
            
            bar.show_progress(++current);
        }
    }
    
    void print_results() {
        std::sort(tests.begin(), tests.end(), [](impl::TestFunc<input_data> i, impl::TestFunc<input_data> j) { return (i.elapsed < j.elapsed);});
        
        for (auto& test : tests) {
            std::cout << test.get_label() << "\t";
            printf("\t%lu (%d)", test.elapsed, test.result);
            std::cout << std::endl;
        }
    }            
};


template<class test_func, class param_tuples>
struct TestHarness {
    typedef typename test_func::input_data input_data;
    
    static Benchmark<input_data>* prepare_benchmark(std::string label) {
        Benchmark<input_data>* bench = new Benchmark<input_data>(param_tuples::total, label);

        impl::GenTests<test_func, typename param_tuples::next, param_tuples::total - 1>::gen(bench->tests);

        return bench;
    }
    
};


}
