cmake_minimum_required(VERSION 3.7)

project(regulus)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA 8.0 REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)

set(COMPUTE_CAPABILITY 61)

list(
  APPEND
  CUDA_NVCC_FLAGS
  
  -gencode arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}
  --expt-extended-lambda
)

add_library(regulus_core INTERFACE)
target_include_directories(
  regulus_core

  INTERFACE
  ${CUDA_INCLUDE_DIR}
  ./include
)

set(
  HEADERS

  ./include/regulus/array.hpp
)

set(
  TESTS

  ./tests/array-tests.cu
)

# Generate the demo/release binary

cuda_add_executable(
  regulus

  ./release/main.cu
  ${HEADERS}
)

target_link_libraries(
  regulus

  regulus_core
)

# Assemble the testing binary

# Find the Catch header and create an interface lib from it
find_path(CATCH_INCLUDE_DIR catch.hpp)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

cuda_add_executable(
  regulus_tests

  ./debug/main.cu
  ${HEADERS}
  ${TESTS}
)

target_link_libraries(
  regulus_tests

  regulus_core
  Catch
)

# see: https://github.com/philsquared/Catch/blob/master/contrib/ParseAndAddCatchTests.cmake
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake.modules/")
#enable_testing()

#include(ParseAndAddCatchTests)
#ParseAndAddCatchTests(regulus_tests)