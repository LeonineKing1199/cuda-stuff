#include "catch.hpp"
#include "globals.hpp"
#include "index_t.hpp"
#include "math/point.hpp"
#include "math/tetra.hpp"
#include "lib/init-ta-and-pa.hpp"
#include "lib/fracture.hpp"
#include "lib/fract-locations.hpp"
#include "lib/nominate.hpp"
#include <thrust/device_vector.h>

auto operator==(tetra const& t1, tetra const& t2) -> bool
{
  return (
    t1.x == t2.x &&
    t1.y == t2.y &&
    t1.z == t2.z &&
    t1.w == t2.w);
}

TEST_CASE("Fracture Routine")
{
  using real = float;
  using thrust::device_vector;
        
  device_vector<point_t<real>> pts;
  pts.reserve(5);
    
  int const num_pts{1};
    
  pts[0] = point_t<real>{2, 2, 2};
  
  pts[1] = point_t<real>{0, 0, 0};
  pts[2] = point_t<real>{9, 0, 0};
  pts[3] = point_t<real>{0, 9, 0};
  pts[4] = point_t<real>{0, 0, 9};
    
  tetra const t{1, 2, 3, 4};
    
  device_vector<index_t> pa{num_pts * 8, -1};
  device_vector<index_t> ta{num_pts * 8, -1};
  device_vector<loc_t>   la{num_pts * 8, -1};
    
  calc_initial_assoc<real><<<bpg, tpb>>>(
    pts.data().get(),
    num_pts,
    t,
    pa.data().get(),
    ta.data().get(),
    la.data().get());
  cudaDeviceSynchronize();
    
  REQUIRE(static_cast<index_t>(pa[0]) == index_t{0});
  REQUIRE(static_cast<index_t>(ta[0]) == index_t{0});
  REQUIRE(static_cast<loc_t>(la[0])   == loc_t{15});
    
  int const assoc_size{1};
  int const num_tetra{1};
       
  device_vector<index_t> fl{assoc_size, -1};
  device_vector<unsigned> nm{num_pts, 0};
  device_vector<tetra> mesh{8 * num_pts};
  mesh[0] = t;
        
  REQUIRE((mesh[0] == tetra{1, 2, 3, 4}));
    
  // nominate the points, calculate the fracture locations
  // and then finally fracture the tetrahedron!
  nominate(assoc_size, pa, ta, la, nm);
  fract_locations(assoc_size, pa, nm, la, fl);      
  fracture(assoc_size, num_tetra, pa, ta, la, nm, fl, mesh);
      
  REQUIRE((num_tetra + static_cast<index_t>(fl[assoc_size - 1]) == 4));
  
  REQUIRE((mesh[0] == tetra{4, 3, 2, 0}));
  REQUIRE((mesh[1] == tetra{1, 3, 4, 0}));
  REQUIRE((mesh[2] == tetra{1, 4, 2, 0}));
  REQUIRE((mesh[3] == tetra{1, 2, 3, 0}));
}
