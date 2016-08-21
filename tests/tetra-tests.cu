#include "test-suite.hpp"
#include "../include/math/tetra.hpp"

__host__ __device__
auto tetra_tests_impl(void) -> void
{
  using real = float;
  using point_t = point_t<real>;
  
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
  
  // Okay, now for location code testing!
  // For any tetrahedron, we have:
  // 4 vertices
  // 4 faces
  // 6 edges
  // an internal region
  // outside the tetrahedron
  // We must make sure our loc routine can accurately
  // solve these
  {
    point_t const a{ 0.0, 0.0, 0.0 };
    point_t const b{ 9.0, 0.0, 0.0 };
    point_t const c{ 0.0, 9.0, 0.0 };
    point_t const d{ 0.0, 0.0, 9.0 };
    
    assert(orient<real>(a, b, c, d) == orientation::positive);
    
    // We should be able to accurately determine all 6 edge intersections
    {
      point_t const e10{ 4.5, 0.0, 0.0 };
      point_t const e20{ 0.0, 4.5, 0.0 };
      point_t const e30{ 0.0, 0.0, 4.5 };
      point_t const e21{ 4.5, 4.5, 0.0 };
      point_t const e31{ 4.5, 0.0, 4.5 };
      point_t const e23{ 0.0, 4.5, 4.5 };
            
      assert((
        eq<real>(det(matrix<real, 4, 4>{ 1, 0, 0, 0,
                                         1, 0, 9, 0,
                                         1, 0, 0, 9,
                                         1, 4.5, 0, 0 }), 364.5)));
                                         
      assert(orient<real>(d, c, b, e10) == orientation::positive);
      assert(orient<real>(a, c, d, e10) == orientation::positive);
      assert(orient<real>(a, d, b, e10) == orientation::zero);
      assert(orient<real>(a, b, c, e10) == orientation::zero);
      
      assert(loc<real>(a, b, c, d, e10) == 3);
      assert(loc<real>(a, b, c, d, e20) == 5);
      assert(loc<real>(a, b, c, d, e30) == 9);
      assert(loc<real>(a, b, c, d, e21) == 6);
      assert(loc<real>(a, b, c, d, e31) == 10);
      assert(loc<real>(a, b, c, d, e23) == 12);
    }
    
    // We should be able to determine all 4 face intersections
    {
      point_t const f321{ 3, 3, 3 };
      point_t const f023{ 0, 4.5, 3 };
      point_t const f031{ 4.5, 0, 3 };
      point_t const f012{ 3, 3, 0 };
      
      assert(loc<real>(a, b, c, d, f321) == 14);
      assert(loc<real>(a, b, c, d, f023) == 13);
      assert(loc<real>(a, b, c, d, f031) == 11);
      assert(loc<real>(a, b, c, d, f012) == 7);
    }
    
    // We should be able to determine all 4 vertex intersections
    {
      point_t const v0 = a;
      point_t const v1 = b;
      point_t const v2 = c;
      point_t const v3 = d;
      
      assert(loc<real>(a, b, c, d, v0) == 1);
      assert(loc<real>(a, b, c, d, v1) == 2);
      assert(loc<real>(a, b, c, d, v2) == 4);
      assert(loc<real>(a, b, c, d, v3) == 8);
    }
    
    // We should be able to determine if a point is inside a tetrahedron
    {
      point_t const p{ 1, 1, 1 };
      
      assert(loc<real>(a, b, c, d, p) == 15);
    }
    
    // We should be able to determine if a point is outside a tetrahedron
    {
      point_t const p{ 3.01, 3.01, 3.01 };
      
      assert(loc<real>(a, b, c, d, p) == 255);
    }
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
