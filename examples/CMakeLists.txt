
add_library(sum_kernels INTERFACE)
target_include_directories(sum_kernels INTERFACE 
                           "${CMAKE_CURRENT_SOURCE_DIR}/sum_kernels"
                           "${CMAKE_CURRENT_SOURCE_DIR}/common")

add_executable(sum_tests_none sum_tests.cpp)
add_executable(sum_tests_sse sum_tests.cpp)
set_target_properties(sum_tests_sse PROPERTIES COMPILE_FLAGS "-msse4.2")
add_executable(sum_tests_avx sum_tests.cpp)
set_target_properties(sum_tests_avx PROPERTIES COMPILE_FLAGS "-mavx2")

target_link_libraries(sum_tests_none avprof sum_kernels)
target_link_libraries(sum_tests_sse avprof sum_kernels)
target_link_libraries(sum_tests_avx avprof sum_kernels)


add_library(mul_kernels INTERFACE)
target_include_directories(mul_kernels INTERFACE 
                           "${CMAKE_CURRENT_SOURCE_DIR}/mul_kernels"
                           "${CMAKE_CURRENT_SOURCE_DIR}/common")

add_executable(mul_tests_none mul_tests.cpp)
add_executable(mul_tests_sse mul_tests.cpp)
set_target_properties(mul_tests_sse PROPERTIES COMPILE_FLAGS "-msse4.2")
add_executable(mul_tests_avx mul_tests.cpp)
set_target_properties(mul_tests_avx PROPERTIES COMPILE_FLAGS "-mavx2")

target_link_libraries(mul_tests_none avprof mul_kernels)
target_link_libraries(mul_tests_sse avprof mul_kernels)
target_link_libraries(mul_tests_avx avprof mul_kernels)

