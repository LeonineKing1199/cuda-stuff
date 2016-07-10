cmake_minimum_required(VERSION 2.8)
find_package(CUDA)

set (CMAKE_CXX_STANDARD 11)
set(CUDA_SEPARABLE_COMPILATION ON)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -rdc=true 
    -lcudart 
    -lcudadevrt 
    -gencode 
    arch=compute_50,code=sm_50 
    -std=c++11 
    -O2 
    --compiler-options "-Wall" 
    --compiler-options "-Wextra"
    )

# Specify include directories
include_directories(
  src
  include
  tests
  )


cuda_add_executable(
  regulus
  main.cu
  include/common.hpp
  src/domain.cu
  src/peano.cu
  tests/test-suite.hpp
  tests/test-suite.cu
  tests/domain-tests.cu)

