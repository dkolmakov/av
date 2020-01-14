#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

namespace av {
namespace impl {
    
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

template<std::size_t... params>
struct KernelParameters {
    typedef impl::CountedParams<0, params...> next;
    static const std::size_t total = next::total;
};

template<class ... kernels>
struct Kernels {
    typedef impl::CountedKernels<0, kernels...> next;
    static const std::size_t total = next::total;
};

template<class kernels, class params>
struct Pairs {
    typedef impl::CountedPairs<typename kernels::next, typename params::next, kernels, params, kernels::total - 1, params::total - 1> next;
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

}

template<std::size_t... params>
struct KernelParameters {
    typedef impl::CountedParams<0, params...> next;
    static const std::size_t total = next::total;
};

template<class ... kernels>
struct Kernels {
    typedef impl::CountedKernels<0, kernels...> next;
    static const std::size_t total = next::total;
};

template<class kernels, class params>
struct Pairs {
    typedef impl::CountedPairs<typename kernels::next, typename params::next, kernels, params, kernels::total - 1, params::total - 1> next;
    static const std::size_t total = kernels::total * params::total;
};

template<class ... elems>
struct Combinations {
    typedef typename impl::CombinationsInt<0, elems...>::val val;
    static const std::size_t total = val::total;
};

}
