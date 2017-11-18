#include <thrust/device_vector.h>

#include "regulus/array.hpp"

#include "regulus/algorithm/redistribute_pts.hpp"

#include <catch.hpp>

TEST_CASE("Point redistribution")
{
  SECTION("should function as expected")
  {
    using point_t = float3;

    auto h_pts = thrust::host_vector<point_t>{};

    // form the root tetrahedron with these 4
    h_pts.push_back(point_t{0, 0, 0});
    h_pts.push_back(point_t{9, 0, 0});
    h_pts.push_back(point_t{0, 9, 0});
    h_pts.push_back(point_t{0, 0, 9});

    // point to insert
    h_pts.push_back(point_t{3, 3, 3});


  }
}