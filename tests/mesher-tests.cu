#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "test-suite.hpp"
#include "../include/mesher.hpp"
#include "../include/domain.hpp"
#include "../include/math/tetra.hpp"
#include "../include/math/point.hpp"

using thrust::host_vector;
using thrust::device_vector;
using std::cout;
using std::endl;

auto mesher_tests(void) -> void
{
  cout << "Beginning mesher tests!\n";
  
  // We should be able to construct a mesh
  {
    using real = float;
    
    // grid of all points in x,y,z 0 through 9
    int const grid_length{10};
    host_vector<point_t<real>> pts{gen_cartesian_domain<real>(grid_length)};
    
    pts.push_back(point_t<real>{0, 0, 0});
    pts.push_back(point_t<real>{grid_length * 3, 0, 0});
    pts.push_back(point_t<real>{0, grid_length * 3, 0});
    pts.push_back(point_t<real>{0, 0, grid_length * 3});
    
    int const a{(int ) pts.size() - 4};
    int const b{(int ) pts.size() - 3};
    int const c{(int ) pts.size() - 2};
    int const d{(int ) pts.size() - 1};
    
    tetra const root_tet{a, b, c, d};
    
    // create mesher instance
    mesher<real> m{pts, root_tet};
    
    // do a quick check that all of our points are actually in the
    // root tetrahedron
    {
      auto const& pa = pts[root_tet.x];
      auto const& pb = pts[root_tet.y];
      auto const& pc = pts[root_tet.z];
      auto const& pd = pts[root_tet.w];
      
      assert(orient<real>(pa, pb, pc, pd) == orientation::positive);
      
      for (int i = 0; i < (grid_length * grid_length * grid_length); ++i) {
        assert(loc<real>(pa, pb, pc, pd, pts[i]) != -1);      
      }
    }
    
    // pray for the best!
    m.triangulate();
  }
  
  cout << "Tests Passed!\n" << endl;
}