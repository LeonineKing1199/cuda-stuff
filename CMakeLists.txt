cmake_minimum_required(VERSION 3.8)

project(regulus)

find_package(CUDA 9.0 REQUIRED)

# Find the Catch header and create an interface lib from it
find_path(CATCH_INCLUDE_DIR catch.hpp)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

# TODO: abstract this out into a toolchain file
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
      --compiler-options "/permissive-"
      -D_SCL_SECURE_NO_WARNINGS
    )

  # this is specific to the Ninja build tool
  set(
    CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:libcmt.lib")

endif()

cuda_add_library(
  regulus_core
  STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/nominate.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/fract_locations.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/fracture.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/mark_nominated_tetra.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/assoc_locations.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/redistribution_cleanup.cu
)

set_target_properties(
  regulus_core

  PROPERTIES
  CXX_STANDARD 14
  CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(
  regulus_core

  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CUDA_INCLUDE_DIR}
)

target_link_libraries(
  regulus_core

  ${CUDA_cudadevrt_LIBRARY}
)

# Assemble the testing binary
cuda_add_executable(
  regulus_tests

  ${CMAKE_CURRENT_SOURCE_DIR}/debug/main.cu

  ${CMAKE_CURRENT_SOURCE_DIR}/tests/array_tests.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/build_root_tetrahedron_tests.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/vector_tests.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/matrix_tests.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/tetra_tests.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/equals.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_assoc_relations_tests.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/numeric_stability_tests.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/nomination_tests.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/bigger_nomination_tests.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/rand_tests.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/span_tests.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/fract_locations_tests.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/fracture_tests.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/mark_nominated_tetra_tests.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/redistribute_pts_tests.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/redistribution_cleanup_tests.cu
)

target_link_libraries(
  regulus_tests

  Catch
  regulus_core
)

# add some CTest functionality as well
# unfortunately, only the Catch binary is run so as far as
# CTest knows, only one test is actually run
# see: https://github.com/philsquared/Catch/blob/master/contrib/ParseAndAddCatchTests.cmake
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake.modules/")

enable_testing()
include(ParseAndAddCatchTests)
ParseAndAddCatchTests(regulus_tests)

# Generate the demo/release binary
cuda_add_executable(
  regulus

  ${CMAKE_CURRENT_SOURCE_DIR}/release/main.cu
)

target_link_libraries(
  regulus

  regulus_core
)