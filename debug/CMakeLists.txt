cmake_minimum_required(VERSION 3.7)
project(regulus)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CUDA_HOST_COMPILATION_CPP ON)
set(CUDA_SEPARABLE_COMPILATION ON)

list(
  APPEND
  CUDA_NVCC_FLAGS
  
  -gencode arch=compute_61,code=sm_61
  --expt-extended-lambda
  )

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
  ../include/maybe-int.hpp
  
  ../include/math/point.hpp
  ../include/math/equals.hpp
  ../include/math/matrix.hpp
  ../include/math/vector.hpp
  ../include/math/tetra.hpp
  ../include/math/rand-int-range.hpp

  ../include/lib/init-ta-and-pa.hpp
  ../include/lib/nominate.hpp
  ../include/lib/set-internal-fract-adjacencies.hpp
  ../include/lib/fracture.hpp
  ../include/lib/mark-nominated-tetra.hpp
  ../include/lib/redistribute-pts.hpp
  
  ../src/point.cu
  ../src/peano.cu
  ../src/rand-int-range.cu
  ../src/nominate.cu
  ../src/fract-locations.cu
  ../src/tetra.cu
  ../src/set-internal-fract-adjacencies.cu
  ../src/fracture.cu
  ../src/mark-nominated-tetra.cu
  
  ../tests/array-tests.cu
  ../tests/matrix-tests.cu
  ../tests/equals.cu
  ../tests/vector-tests.cu
  ../tests/tetra-tests.cu
  ../tests/domain-tests.cu
  ../tests/nomination-tests.cu
  ../tests/fract-location-tests.cu
  ../tests/index_t-tests.cu
  ../tests/set-internal-fract-adjacencies.cu
  ../tests/fracture-tests.cu
  ../tests/redistribute-pts-tests.cu
  )
  
target_link_libraries(
  regulus
  
  ${CUDA_cudadevrt_LIBRARY}
  )
   
