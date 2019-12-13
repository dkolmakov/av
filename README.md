# av

Header-only auto-vectorization (av) library for primitive arithmetic operations over arrays.

The goal of the project is to provide the best arithmetic kernels for the given architecture.

Library is auto-generated with the help of cmake building system.
It can be included to the destination project simply by adding `include_sudirectory(./thirdparty/av)`.

## Requirements

- Doesn't contain cpp files
- Performs auto adjustment during cmake build
- Provides a way to perform specialization against user-defined function
