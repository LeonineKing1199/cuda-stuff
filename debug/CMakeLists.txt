cmake_minimum_required(VERSION 3.7)
project(regulus)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CUDA_HOST_COMPILATION_CPP ON)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_64_BIT_DEVICE_CODE ON)

set(
  CUDA_NVCC_FLAGS
  
  -gencode arch=compute_61,code=sm_61
  --expt-extended-lambda
  )
                                 
link_directories(
  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/lib/x64"
  )        
        
include_directories(
  ../include
  )
   
cuda_add_executable(
  regulus
  
  main.cu
  
  ../include/array.hpp
  ../include/enable_if.hpp
  ../include/domain.hpp
  ../include/size_type.hpp
  ../include/index_t.hpp
  
  ../include/math/point.hpp
  ../include/math/equals.hpp
  ../include/math/matrix.hpp
  ../include/math/vector.hpp
  ../include/math/tetra.hpp

  ../include/lib/init-ta-and-pa.hpp
  ../include/lib/nominate.hpp
  
  ../src/point.cu
  ../src/peano.cu
  ../src/nominate.cu
  
  ../tests/array-tests.cu
  ../tests/matrix-tests.cu
  ../tests/equals.cu
  ../tests/vector-tests.cu
  ../tests/tetra-tests.cu
  ../tests/domain-tests.cu
  ../tests/nomination-tests.cu
  )
  
target_link_libraries(
  regulus
  
  cudadevrt
  cudart
  )
   
