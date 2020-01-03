cmake_minimum_required(VERSION 3.2)
project(av_profiler VERSION 0.4.0 LANGUAGES CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_FLAGS "-std=c++14 -fPIC -fpic -Wall -Wextra -pedantic -O3")

add_library(avprof INTERFACE)
target_include_directories(avprof INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(examples)

install(TARGETS avprof DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

