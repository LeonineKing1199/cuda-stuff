cmake_minimum_required(VERSION 3.7)

project(regulus)

find_package(CUDA 9.0 REQUIRED)

# Find the Catch header and create an interface lib from it
find_path(CATCH_INCLUDE_DIR catch.hpp)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

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
  ./src/nominate.cu
  ./src/fract_locations.cu
  ./src/fracture.cu

  ./include/regulus/array.hpp
  ./include/regulus/globals.hpp
  ./include/regulus/is_point.hpp
  ./include/regulus/matrix.hpp
  ./include/regulus/mesh.hpp
  ./include/regulus/orientation.hpp
  ./include/regulus/point.hpp
  ./include/regulus/point_traits.hpp
  ./include/regulus/type_traits.hpp
  ./include/regulus/utils
  ./include/regulus/vector.hpp
  ./include/regulus/loc.hpp
  ./include/regulus/tetra.hpp

  ./include/regulus/algorithm/build_root_tetrahedron.hpp
  ./include/regulus/algorithm/insphere.hpp
  ./include/regulus/algorithm/location.hpp
  ./include/regulus/algorithm/make_assoc_relations.hpp
  ./include/regulus/algorithm/nominate.hpp
  ./include/regulus/algorithm/orient.hpp
  ./include/regulus/algorithm/fract_locations.hpp
  ./include/regulus/algorithm/fracture.hpp

  ./include/regulus/utils/dist_from_plane.hpp
  ./include/regulus/utils/equals.hpp
  ./include/regulus/utils/gen_cartesian_domain.hpp
  ./include/regulus/utils/make_point.hpp
  ./include/regulus/utils/numeric_limits.hpp
  ./include/regulus/utils/round_to.hpp
  ./include/regulus/utils/make_rand_range.hpp

  ./include/regulus/views/span.hpp
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
  ./include
  ${CUDA_INCLUDE_DIR}
)

target_link_libraries(
  regulus_core

  ${CUDA_cudadevrt_LIBRARY}
)

# Assemble the testing binary
cuda_add_executable(
  regulus_tests

  ./debug/main.cu

  ./tests/array_tests.cu
  ./tests/build_root_tetrahedron_tests.cu
  ./tests/vector_tests.cu
  ./tests/matrix_tests.cu
  ./tests/tetra_tests.cu
  ./tests/equals.cu
  ./tests/make_assoc_relations_tests.cu
  ./tests/numeric_stability_tests.cu
  ./tests/nomination_tests.cu
  ./tests/bigger_nomination_tests.cu
  ./tests/rand_tests.cu
  ./tests/span_tests.cu
  ./tests/fract_locations_tests.cu
  ./tests/fracture_tests.cu
)

target_link_libraries(
  regulus_tests

  Catch
  regulus_core
)

# Generate the demo/release binary
cuda_add_executable(
  regulus

  ./release/main.cu
)

target_link_libraries(
  regulus

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