cmake_minimum_required(VERSION 3.9)
project(lab1)

set(CMAKE_CXX_STANDARD 11)

include_directories(/opt/intel/mkl/include)

add_executable(lab1-c1-c2 lab1-c1-c2.c)
add_executable(lab1-c5-c6 lab1-c5-c6.c)
add_executable(sandbox sandbox.c)
