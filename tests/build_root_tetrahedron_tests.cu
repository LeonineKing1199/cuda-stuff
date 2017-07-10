#include <catch.hpp>

#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "regulus/point_t.hpp"
#include "regulus/algorithm/build_root_tetrahedron.hpp"
#include "regulus/utils/gen_cartesian_domain.hpp"

namespace
{
namespace T = thrust;
namespace R = regulus;

using point_t = R::point_t<double>;

auto operator<<(std::ostream& os, point_t const p) -> std::ostream&
{
  os << "{ " << p.x << ", " << p.y << ", " << p.z << " }";
  return os;
}

} // anonymous namespace

TEST_CASE("Building the all-encompassing global tetrahedron... ")
{
  auto const grid_length = size_t{9};

  // build the host point set first
  auto h_pts = T::host_vector<point_t>{};
  h_pts.resize(grid_length * grid_length * grid_length);
  auto begin = h_pts.begin();
  R::gen_cartesian_domain<point_t>(grid_length, begin);

  // copy to device and call relevant function for testing
  auto d_pts = thrust::device_vector<point_t>{h_pts};
  auto const vertices = 
    R::build_root_tetrahedron<point_t>(
      d_pts.begin(), 
      d_pts.end());

}