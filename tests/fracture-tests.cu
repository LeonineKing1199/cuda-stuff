#include <thrust/device_vector.h>
#include <bitset>

#include "test-suite.hpp"
#include "../include/globals.hpp"
#include "../include/lib/fracture.hpp"
#include "../include/math/tetra.hpp"
#include "../include/math/point.hpp"
#include "../include/lib/calc-ta-and-pa.hpp"
#include "../include/lib/nominate.hpp"
#include "../include/lib/fract-locations.hpp"

auto operator==(tetra const& t1, tetra const& t2) -> bool
{
  return t1.x == t2.x && t1.y == t2.y && t1.z == t2.z && t1.w == t2.w;    
}

auto fracture_tests(void) -> void
{
  std::cout << "Beginning fracture tests!" << std::endl;
  
  // We should be able to fracture a tetrahedron
  {
    using real = float;
        
    thrust::device_vector<point_t<real>> pts;
    pts.reserve(5);
    
    int const num_pts = 1;
    
    pts[0] = point_t<real>{2, 2, 2};
    
    pts[1] = point_t<real>{0, 0, 0};
    pts[2] = point_t<real>{9, 0, 0};
    pts[3] = point_t<real>{0, 9, 0};
    pts[4] = point_t<real>{0, 0, 9};
    
    tetra const t{1, 2, 3, 4};
    
    thrust::device_vector<int> pa{num_pts * 8, -1};
    thrust::device_vector<int> ta{num_pts * 8, -1};
    thrust::device_vector<int> la{num_pts * 8, -1};
    
    calc_initial_assoc<real><<<bpg, tpb>>>(
      pts.data().get(),
      num_pts,
      t,
      pa.data().get(),
      ta.data().get(),
      la.data().get());
    cudaDeviceSynchronize();
    
    assert(pa[0] == 0);
    assert(ta[0] == 0);
    assert(la[0] == 15);
    
    int const assoc_size = 1;
    int const num_tetra = 1;
    
    thrust::device_vector<int> nm_ta{num_tetra, -1};
    thrust::device_vector<int> nm{num_pts, 1};
    
    nominate<<<bpg, tpb>>>(
      assoc_size,
      ta.data().get(),
      pa.data().get(),
      nm_ta.data().get(),
      nm.data().get());
    
    repair_nm_ta<<<bpg, tpb>>>(
      assoc_size,
      pa.data().get(),
      ta.data().get(),
      nm.data().get(),
      nm_ta.data().get());
      
    cudaDeviceSynchronize();
    
    thrust::device_vector<int> fl{assoc_size, -1};
    
    fract_locations(
      pa.data().get(),
      nm.data().get(),
      la.data().get(),
      assoc_size,
      fl.data().get());
      
    thrust::device_vector<tetra> mesh;
    mesh.reserve(8 * num_pts);
    mesh[0] = t;
        
    assert((mesh[0] == tetra{1, 2, 3, 4}));
      
    fracture<<<bpg, tpb>>>(
      assoc_size,
      num_tetra,
      pa.data().get(),
      ta.data().get(),
      la.data().get(),
      nm.data().get(),
      fl.data().get(),
      mesh.data().get());
      
    cudaDeviceSynchronize();
            
    
    assert((
      num_tetra +
      fl[assoc_size - 1] +
      nm[pa[assoc_size - 1]] * (std::bitset<4>{la[assoc_size - 1]}.count() - 1) == 4));
    
    assert((mesh[0] == tetra{4, 3, 2, 0}));
    assert((mesh[1] == tetra{1, 3, 4, 0}));
    assert((mesh[2] == tetra{1, 4, 2, 0}));
    assert((mesh[3] == tetra{1, 2, 3, 0}));
  }
  
  std::cout << "Tests Passed!\n" << std::endl;
}
