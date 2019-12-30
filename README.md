# av

Header-only auto-vectorization (av) library for primitive arithmetic operations over arrays.

The goal of the project is to provide the best arithmetic kernels for the given architecture.

Library is auto-generated with the help of cmake building system.
It can be included to the destination project simply by adding `include_sudirectory(./thirdparty/av)`.

## Requirements

- Doesn't contain cpp files
- Performs auto adjustment during cmake build
- Provides a way to perform specialization against user-defined function


## Description

- `kernel` - an elementary arithmetic operation, has `core` template structure and get_label() function
- `test_function` - user defined structure with fixed interface containing function `compute` and structure `input_data`. `compute` can be built using one or more `kernels`.
- `benchmark` - automatically generated set of `kernel_tests`, one for each specific `kernel`.
- `kernel_tests` - a set of `test_functions` built using a specialized version of kernels, one for each specific `kernel_parameter`.
- `kernel_parameter` - a value which specializes kernel by characteristics depending on the hardware properties.
