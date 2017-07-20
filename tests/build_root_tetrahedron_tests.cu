#include <iostream>
#include <iterator>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/uninitialized_copy.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>

#include "regulus/point_t.hpp"
#include "regulus/algorithm/build_root_tetrahedron.hpp"
#include "regulus/utils/gen_cartesian_domain.hpp"
#include "regulus/algorithm/location.hpp"

#include <catch.hpp>

namespace
{
  namespace T = thrust;
  namespace R = regulus;

  using point_t = R::point_t<float>;

  template <typename Point>
  struct relative_loc : public T::unary_function<Point const, bool>
  {
    R::array<Point, 4> vertices;

    relative_loc(void) = delete;
    relative_loc(R::array<Point, 4> const vtx)
    {
      T::uninitialized_copy(
        T::seq,
        vtx.begin(), vtx.end(),
        vertices.begin());
    }

    __host__ __device__
    auto operator()(Point const p) -> bool
    {
      return (
        16 > loc(
          vertices[0],
          vertices[1],
          vertices[2],
          vertices[3],
          p));
    }
  };
} // anonymous namespace

TEST_CASE("Building the all-encompassing global tetrahedron... ")
{
  auto const grid_length = size_t{9};

  // build the host point set first
  auto h_pts = T::host_vector<point_t>{};
  h_pts.reserve(grid_length * grid_length * grid_length);

  {
    auto output_begin = std::back_inserter(h_pts);
    R::gen_cartesian_domain<point_t>(grid_length, output_begin);
  }
  
  // copy to device and call relevant function for testing
  auto d_pts = thrust::device_vector<point_t>{h_pts};
  auto const vertices = 
    R::build_root_tetrahedron<point_t>(
      d_pts.begin(), 
      d_pts.end());

  auto const all_are_contained = T::all_of(
      d_pts.begin(), d_pts.end(), 
      relative_loc<point_t>{vertices});

  REQUIRE(all_are_contained);
}