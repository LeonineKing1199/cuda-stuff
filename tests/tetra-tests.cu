#include "test-suite.hpp"
#include "../include/math/tetra.hpp"

__host__ __device__
auto tetra_tests_impl(void) -> void
{
  // We should be able to determine the orientation
  // of a tetrahedron correctly
  {
  
  }
}

__global__
void tetra_tests_kernel(void)
{
  tetra_tests_impl();
}

auto tetra_tests(void) -> void
{
  std::cout << "Beginning tetra tests!" << std::endl;
  
  tetra_tests_impl();
  
  tetra_tests_kernel<<<1, 256>>>();
  cudaDeviceSynchronize();
  
  std::cout << "All tests passed\n" << std::endl;
}
