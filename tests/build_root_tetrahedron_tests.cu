#include <iostream>
#include <iterator>

// not sure about Linux but currently, I get a warning
// from nvcc about conversions in the interal Thrust libs
// currently only msvc seems to be the culprit, this is a
// rudimentary fix to filter the warning from the build
// log
#ifdef _MSC_VER
#pragma warning(push)

// thrust/iterator/iterator_adaptor.h(203)
// : warning C4244: '+=': conversion from '__int64' to 'int', possible loss of data
#pragma warning(disable: 4244)

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/uninitialized_copy.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>

#pragma warning(pop)

#elif

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/uninitialized_copy.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>

#endif

#include "regulus/point_t.hpp"
#include "regulus/algorithm/build_root_tetrahedron.hpp"
#include "regulus/utils/gen_cartesian_domain.hpp"
#include "regulus/algorithm/location.hpp"

#include <catch.hpp>

namespace
{
  using point_t = regulus::point_t<float>;

  template <typename Point>
  struct relative_loc : public thrust::unary_function<Point const, bool>
  {
    regulus::array<Point, 4> vertices;

    relative_loc(void) = delete;
    relative_loc(regulus::array<Point, 4> const vtx)
    {
      thrust::uninitialized_copy(
        thrust::seq,
        vtx.begin(), vtx.end(),
        vertices.begin());
    }

    __host__ __device__
    auto operator()(Point const p) -> bool
    {
      auto const tmp = loc(
        vertices[0],
        vertices[1],
        vertices[2],
        vertices[3],
        p);

      return tmp < 16;
    }
  };
} // anonymous namespace

TEST_CASE("Building the all-encompassing global tetrahedron... ")
{
  auto const grid_length = size_t{9};

  // build the host point set first
  auto h_pts = thrust::host_vector<point_t>{};
  h_pts.reserve(grid_length * grid_length * grid_length);

  {
    auto output_begin = std::back_inserter(h_pts);
    regulus::gen_cartesian_domain<point_t>(grid_length, output_begin);
  }
  
  // copy to device and call relevant function for testing
  auto          d_pts = thrust::device_vector<point_t>{h_pts};
  auto const vertices =
    regulus::build_root_tetrahedron<point_t>(
      d_pts.begin(), 
      d_pts.end());

  // as a precaution, make sure the root tetrahedron we're proposing
  // is positively oriented
  REQUIRE(
    (orient(
      vertices[0], 
      vertices[1], 
      vertices[2], 
      vertices[3]) == regulus::orientation::positive));

  // and now finally ensure that each point of cartesian grid
  // is contained by the proposed all-encompassing tetrahedron
  auto const all_are_contained = thrust::all_of(
    d_pts.begin(), d_pts.end(), 
    relative_loc<point_t>{vertices});

  REQUIRE(all_are_contained);
}