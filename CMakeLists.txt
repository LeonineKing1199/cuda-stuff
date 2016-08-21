cmake_minimum_required(VERSION 3.2)

project(regulus)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CUDA_HOST_COMPILATION_CPP ON)
set(CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter")

# Pass options to NVCC
list(APPEND
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -gencode arch=compute_50,code=sm_50 
    -std=c++11 
    -O3
    )

cuda_add_executable(
  regulus
  
  main.cu

  include/domain.hpp
  include/timer.hpp
  include/mesher.hpp
  include/array.hpp
  include/globals.hpp
  
  include/math/math.hpp
  include/math/point.hpp
  include/math/matrix.hpp
  include/math/equals.hpp
  include/math/tetra.hpp
  
  include/lib/calc-ta-and-pa.hpp
  include/lib/nominate.hpp
  
  tests/test-suite.hpp
  
  src/point.cu
  src/peano.cu
  
  tests/test-suite.cu
  tests/domain-tests.cu
  tests/mesher-tests.cu
  tests/math-tests.cu
  tests/matrix-tests.cu
  tests/array-tests.cu
  tests/tetra-tests.cu
  tests/nomination-tests.cu
  )

target_link_libraries(
  regulus
  ${CUDA_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  )
