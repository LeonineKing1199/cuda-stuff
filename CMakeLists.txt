cmake_minimum_required(VERSION 3.7)
project(regulus)

enable_testing()
find_package(GTest REQUIRED)

find_package(CUDA REQUIRED)

set(CUDA_64_BIT_DEVICE_CODE ON)

set(
  CUDA_NVCC_FLAGS
  
  -gencode arch=compute_61,code=sm_61
  )
                                 
link_directories(
  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/lib/x64"
  )        
        
include_directories(
  include
  src
  tests
  "C:/Users/christian/gtest/googletest/googletest/include"
  )
   
cuda_add_executable(
  regulus
  
  main.cu
  )
  
target_link_libraries(
  regulus
  
  cudadevrt
  "C:/Users/christian/gtest/build/googlemock/gtest/Release/gtest_main.lib"
  "C:/Users/christian/gtest/build/googlemock/gtest/Release/gtest.lib"
  )
    
add_test(FactorialTest regulus)  
