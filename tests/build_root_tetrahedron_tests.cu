#include <catch.hpp>

#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "regulus/point_t.hpp"
#include "regulus/algorithm/build_root_tetrahedron.hpp"
#include "regulus/utils/gen_cartesian_domain.hpp"

namespace T = thrust;
namespace R = regulus;

TEST_CASE("Building the all-encompassing global tetrahedron... ")
{
  auto const grid_length = size_t{9};

  using point_t = R::point_t<double>;

  auto h_pts = T::host_vector<point_t>{};
  h_pts.resize(grid_length * grid_length * grid_length);

  auto begin = h_pts.begin();
  R::gen_cartesian_domain<point_t>(grid_length, begin);

}