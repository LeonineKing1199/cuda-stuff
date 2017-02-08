#include "catch.hpp"
#include "globals.hpp"
#include "index_t.hpp"
#include "array.hpp"
#include "math/point.hpp"
#include "math/tetra.hpp"
#include "lib/init-ta-and-pa.hpp"
#include "lib/fracture.hpp"
#include "lib/fract-locations.hpp"
#include "lib/nominate.hpp"

#include <thrust/device_vector.h>
#include <iostream>

TEST_CASE("Fracture Routine")
{
  using real = float;
  using thrust::device_vector;
  
  // reserve space for 5 points: 
  // 4 for tetrahedral vertices      
  // 1 for point insertion
  device_vector<point_t<real>> pts{ 5, point_t<real>{ -1, -1, -1  } };
  
  // num points to insert  
  int const num_pts = 1;
    
  pts[0] = point_t<real>{ 2, 2, 2 };
  
  pts[1] = point_t<real>{ 0, 0, 0 };
  pts[2] = point_t<real>{ 9, 0, 0 };
  pts[3] = point_t<real>{ 0, 9, 0 };
  pts[4] = point_t<real>{ 0, 0, 9 };
    
  tetra const t = { 1, 2, 3, 4 };
    
  device_vector<index_t> pa{ num_pts * 8, -1 };
  device_vector<index_t> ta{ num_pts * 8, -1 };
  device_vector<loc_t>   la{ num_pts * 8, -1 };

  calc_initial_assoc<real><<<bpg, tpb>>>(
    pts.data().get(),
    num_pts,
    t,
    pa.data().get(),
    ta.data().get(),
    la.data().get());

  cudaDeviceSynchronize();
  
  // point 0 is inside tetrahedron 0 with location code 15  
  REQUIRE(static_cast<index_t>(pa[0]) == index_t{0});
  REQUIRE(static_cast<index_t>(ta[0]) == index_t{0});
  REQUIRE(static_cast<loc_t>(la[0])   == loc_t{15});
        
  size_t const assoc_size = 1;
  size_t const num_tetra  = 1;
       
  device_vector<index_t>  fl{ assoc_size, -1 };
  device_vector<unsigned> nm{ num_pts, 0 };

  size_t const num_est_tetra = 8 * num_pts;

  device_vector<tetra> mesh{ num_est_tetra, tetra{ -1, -1, -1, -1 } };
  mesh[0] = t;
  
  tetra const expected_tetrahedron = { 1, 2, 3, 4 };
  REQUIRE(mesh[0] == expected_tetrahedron);
  
  device_vector<adjacency> adjacency_relations{ num_est_tetra, adjacency{ -1, -1, -1, -1 } };

  // nominate the points, calculate the fracture locations
  // and then finally fracture the tetrahedron!
  nominate(assoc_size, pa, ta, la, nm);
  fract_locations(assoc_size, pa, nm, la, fl);      
  fracture(assoc_size, num_tetra, pa, ta, la, nm, fl, mesh, adjacency_relations);

  cudaDeviceSynchronize();

  REQUIRE((num_tetra + static_cast<index_t>(fl[assoc_size - 1]) == 4));
  
  array<tetra, 4> expected_tets = {
    tetra{ 4, 3, 2, 0 },
    tetra{ 1, 3, 4, 0 },
    tetra{ 1, 4, 2, 0 },
    tetra{ 1, 2, 3, 0 }
  };

  for (decltype(expected_tets.size()) i = 0; i < expected_tets.size(); ++i) {
    REQUIRE(mesh[i] == expected_tets[i]);
  }

  size_t const face_idx[4][3] = {
    { 3, 2, 1 },   // face 0
    { 0, 2, 3 },   // face 1
    { 0, 3, 1 },   // face 2
    { 0, 1, 2 }    // face 3
  };

  for (size_t i = 0; i < 4; ++i) {
    auto const ar = static_cast<adjacency>(adjacency_relations[i]);

    REQUIRE(mesh[ar.x] == mesh[face_idx[i][0]]);
    REQUIRE(mesh[ar.y] == mesh[face_idx[i][1]]);
    REQUIRE(mesh[ar.z] == mesh[face_idx[i][2]]);
    REQUIRE(ar.w       == -1);
  }
}
