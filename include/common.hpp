#pragma once

#include <complex>

namespace av {
#define force_inline inline __attribute__((always_inline))
    
#ifdef __AVX512F__
    const std::string inst_set = "AVX512F";
    constexpr std::size_t SIMD_REG_SIZE = 64;
#elif __AVX__
    const std::string inst_set = "AVX";
    constexpr std::size_t SIMD_REG_SIZE = 32;
#elif __SSE4_1__
    const std::string inst_set = "SSE4.1";
    constexpr std::size_t SIMD_REG_SIZE = 16;
#else
    const std::string inst_set = "Default";
    constexpr std::size_t SIMD_REG_SIZE = 0;
#endif


template<class T>
struct CalculationTask {
  std::string label;
  std::complex<T> (*func)(std::complex<T> *arr, std::size_t count);    
};

}
