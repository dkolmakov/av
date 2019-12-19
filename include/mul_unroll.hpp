#pragma once

#include <complex>

#include "common.hpp"

namespace mul_unroll {
    
namespace implementation {

    template <class T, std::size_t index>
    struct unpack_complex;
    
    template <class T>
    struct unpack_complex<T, 0> {
        static force_inline void compute(T *real, T *imag, std::complex<T> *arr) {
            real[0] = arr[0].real();
            imag[0] = arr[0].imag();
        }
    };
    
    template <class T, std::size_t index>
    struct unpack_complex {
        static force_inline void compute(T *real, T *imag, std::complex<T> *arr) {
            real[index] = arr[index].real();
            imag[index] = arr[index].imag();
            unpack_complex<T, index - 1>::compute(real, imag, arr);
        }
    };

    
    template <class T, std::size_t index>
    struct mul_real_real;
    
    template <class T>
    struct mul_real_real<T, 0> {
        static force_inline void compute(T *rr, T *real) {
            rr[0] = real[0] * real[1];
        }
    };
    
    template <class T, std::size_t index>
    struct mul_real_real {
        static force_inline void compute(T *rr, T *real) {
            rr[index] = real[index * 2] * real[index * 2 + 1];
            mul_real_real<T, index - 1>::compute(rr, real);
        }
    };


    template <class T, std::size_t index>
    struct mul_real_imag;
    
    template <class T>
    struct mul_real_imag<T, 0> {
        static force_inline void compute(T *ri, T *real, T *imag) {
            ri[0] = real[0] * imag[1];
        }
    };
    
    template <class T, std::size_t index>
    struct mul_real_imag {
        static force_inline void compute(T *ri, T *real, T *imag) {
            ri[index] = real[index * 2] * imag[index * 2 + 1];
            mul_real_imag<T, index - 1>::compute(ri, real, imag);
        }
    };


    template <class T, std::size_t index>
    struct mul_imag_real;
    
    template <class T>
    struct mul_imag_real<T, 0> {
        static force_inline void compute(T *ir, T *real, T *imag) {
            ir[0] = imag[0] * real[1];
        }
    };
    
    template <class T, std::size_t index>
    struct mul_imag_real {
        static force_inline void compute(T *ir, T *real, T *imag) {
            ir[index] = imag[index * 2] * real[index * 2 + 1];
            mul_imag_real<T, index - 1>::compute(ir, real, imag);
        }
    };
    
    
    template <class T, std::size_t index>
    struct mul_imag_imag;
    
    template <class T>
    struct mul_imag_imag<T, 0> {
        static force_inline void compute(T *ii, T *imag) {
            ii[0] = imag[0] * imag[1];
        }
    };
    
    template <class T, std::size_t index>
    struct mul_imag_imag {
        static force_inline void compute(T *ii, T *imag) {
            ii[index] = imag[index * 2] * imag[index * 2 + 1];
            mul_imag_imag<T, index - 1>::compute(ii, imag);
        }
    };

    
    template <class T, std::size_t index>
    struct calc_real;
    
    template <class T>
    struct calc_real<T, 0> {
        static force_inline void compute(T *real, T *rr, T *ii) {
            real[0] = rr[0] - ii[0];
        }
    };
    
    template <class T, std::size_t index>
    struct calc_real {
        static force_inline void compute(T *real, T *rr, T *ii) {
            real[index] = rr[index] - ii[index];
            calc_real<T, index - 1>::compute(real, rr, ii);
        }
    };

    
    template <class T, std::size_t index>
    struct calc_imag;
    
    template <class T>
    struct calc_imag<T, 0> {
        static force_inline void compute(T *imag, T *ri, T *ir) {
            imag[0] = ri[0] + ir[0];
        }
    };
    
    template <class T, std::size_t index>
    struct calc_imag {
        static force_inline void compute(T *imag, T *ri, T *ir) {
            imag[index] = ri[index] + ir[index];
            calc_imag<T, index - 1>::compute(imag, ri, ir);
        }
    };


    template <class T, std::size_t chunk_size, std::size_t parity_checker>
    struct calc_rec;
    
    template <class T>
    struct calc_rec<T, 2, 0> {
        static force_inline std::complex<T> compute(T *real, T *imag) {
            T res_real[1];
            T rr[1];
            T ii[1];
            
            T res_imag[1];
            T ri[1];
            T ir[1];
            
            mul_real_real<T, 0>::compute(rr, real);
            mul_imag_imag<T, 0>::compute(ii, imag);
            calc_real<T, 0>::compute(res_real, rr, ii);

            mul_real_imag<T, 0>::compute(ri, real, imag);
            mul_imag_real<T, 0>::compute(ir, real, imag);
            calc_imag<T, 0>::compute(res_imag, ri, ir);
            
            return std::complex<T>(res_real[0], res_imag[0]);
        }
    };
    
    template <class T, std::size_t chunk_size>
    struct calc_rec<T, chunk_size, 0> {
        static force_inline std::complex<T> compute(T *real, T *imag) {
            T res_real[chunk_size / 2];
            T rr[chunk_size / 2];
            T ii[chunk_size / 2];
            
            T res_imag[chunk_size / 2];
            T ri[chunk_size / 2];
            T ir[chunk_size / 2];
            
            mul_real_real<T, chunk_size / 2 - 1>::compute(rr, real);
            mul_imag_imag<T, chunk_size / 2 - 1>::compute(ii, imag);
            calc_real<T, chunk_size / 2 - 1>::compute(res_real, rr, ii);

            mul_real_imag<T, chunk_size / 2 - 1>::compute(ri, real, imag);
            mul_imag_real<T, chunk_size / 2 - 1>::compute(ir, real, imag);
            calc_imag<T, chunk_size / 2 - 1>::compute(res_imag, ri, ir);
            
            return calc_rec<T, chunk_size / 2, (chunk_size / 2) % 2>::compute(res_real, res_imag);
        }
    };

    template <class T, std::size_t chunk_size, std::size_t parity_checker>
    struct calc_rec {
        static force_inline std::complex<T> compute(T *real, T *imag) {
            std::complex<T> res = calc_rec<T, chunk_size - 1, 0>::compute(real, imag);

            unpack_complex<T, 0>::compute(&real[chunk_size - 2], &imag[chunk_size - 2], &res);

            return calc_rec<T, 2, 0>::compute(&real[chunk_size - 2], &imag[chunk_size - 2]);
        }
    };
    
    template <class T, std::size_t chunk_size>
    struct chunk_mul;
    
    template <class T>
    struct chunk_mul<T, 1> {
        static force_inline std::complex<T> compute(std::complex<T> *arr) {
            return arr[0];
        }
    };
    
    template <class T, std::size_t chunk_size>
    struct chunk_mul {
        static force_inline std::complex<T> compute(std::complex<T> *arr) {
            T real[chunk_size];
            T imag[chunk_size];

            unpack_complex<T, chunk_size - 1>::compute(real, imag, arr);
            
            return calc_rec<T, chunk_size, chunk_size % 2>::compute(real, imag);
        }
    };

    template <class T, std::size_t chunk_size>
    struct mul {
        static force_inline std::complex<T> compute(std::complex<T> *arr, std::size_t count) {
            // Specialized implementation
            std::complex<T> result(1,0);
            
            asm volatile ("nop;nop;nop;");
            // Sum by chunks
            std::size_t i = 0;
            while (i + chunk_size < count) {
                result *= chunk_mul<T, chunk_size>::compute(arr + i);
                i += chunk_size;
            }
            asm volatile ("nop;nop;nop;");
            
            // Add the remainder
            for (; i < count; i++) {
                result *= arr[i];
            }
            
            return result;
        }
    };
    
}

    template<class T, std::size_t chunk_size>
    static std::complex<T> mul(std::complex<T> *arr, std::size_t count) {
        return implementation::mul<T, chunk_size>::compute(arr, count);
    }

    template<class T, std::size_t chunk_size>
    struct ToTest {
        static std::complex<T> to_test(std::complex<T> *arr, std::size_t count) {
            return mul<T, chunk_size>(arr, count);
        }
    };

}


