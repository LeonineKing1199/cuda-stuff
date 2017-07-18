cmake_minimum_required(VERSION 3.7)

project(regulus)

find_package(CUDA 8.0 REQUIRED)

# Find the Catch header and create an interface lib from it
find_path(CATCH_INCLUDE_DIR catch.hpp)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

set(CMAKE_CXX_STANDARD 11)
set(CUDA_SEPARABLE_COMPILATION ON)
set(COMPUTE_CAPABILITY 61)

list(
  APPEND
  CUDA_NVCC_FLAGS
  
  -gencode arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}
  --expt-extended-lambda
)

# MSVC + Ninja-specific build details
if (MSVC)
  list(
    APPEND
    CUDA_NVCC_FLAGS
    
      # this is specific to the Ninja build tool
      --compiler-options "/FS"
      -D_SCL_SECURE_NO_WARNINGS
    )

  # this is specific to the Ninja build tool
  set(
    CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:libcmt.lib")

endif()


add_library(regulus_core INTERFACE)
target_include_directories(regulus_core INTERFACE ./include)


# Generate the demo/release binary
cuda_add_executable(
  regulus

  ./release/main.cu
)

target_link_libraries(
  regulus

  regulus_core
)

# Assemble the testing binary
set(
  TESTS

  ./tests/array_tests.cu
  ./tests/build_root_tetrahedron_tests.cu
  ./tests/vector_tests.cu
  ./tests/matrix_tests.cu
)

cuda_add_executable(
  regulus_tests

  ./debug/main.cu
  ${TESTS}
)

target_link_libraries(
  regulus_tests

  Catch
  regulus_core
)