#include "test-suite.hpp"
#include "../include/math/tetra.hpp"

__host__ __device__
auto tetra_tests_impl(void) -> void
{
  using real = float;
  using point_t = reg::point_t<real>;
  
  // We should be able to determine the orientation
  // of a tetrahedron correctly
  {    
    point_t const a{ 0.0, 0.0, 0.0 };
    point_t const b{ 9.0, 0.0, 0.0 };
    point_t const c{ 0.0, 9.0, 0.0 };
    point_t const d{ 0.0, 0.0, 9.0 };
    
    assert(orient<real>(a, b, c, d) == orientation::positive);
    
    point_t const e{ 3.0, 3.0, 0.0 };
    assert(orient<real>(a, b, c, e) == orientation::zero);
    
    point_t const f{ 3.0, 3.0, -3.0 };
    assert(orient<real>(a, b, c, f) == orientation::negative);
  }
  
  // insphere stuff should work as well
  {
    point_t const a{ 0.0, 0.0, 0.0 };
    point_t const b{ 9.0, 0.0, 0.0 };
    point_t const c{ 0.0, 9.0, 0.0 };
    point_t const d{ 0.0, 0.0, 9.0 };
    
    // asserting logical basis for insphere results
    assert(orient<real>(a, b, c, d) == orientation::positive);
    
    point_t const x{ 3.0, 3.0, 3.0 };
    assert(insphere<real>(a, b, c, d, x) == orientation::negative);
    
    point_t const y{ 1000.0, 1000.0, 1000.0 };
    assert(insphere<real>(a, b, c, d, y) == orientation::positive);
    
    point_t const z = b;
    assert(insphere<real>(a, b, c, d, z) == orientation::zero);
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
