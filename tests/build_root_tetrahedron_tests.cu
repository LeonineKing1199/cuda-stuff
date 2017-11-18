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

#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <thrust/uninitialized_copy.h>

#pragma warning(pop)

#elif

#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <thrust/uninitialized_copy.h>

#endif

#include "regulus/point.hpp"

#include "regulus/views/span.hpp"

#include "regulus/utils/gen_cartesian_domain.hpp"

#include "regulus/algorithm/orient.hpp"
#include "regulus/algorithm/location.hpp"
#include "regulus/algorithm/build_root_tetrahedron.hpp"

#include <catch.hpp>

using std::size_t;
using std::ptrdiff_t;

using regulus::span;
using regulus::array;

using point_t = double3;

auto pts_are_contained(
  array<point_t, 4>   const vtx,
  span<point_t const> const pts) -> bool
{
  auto const tmp = thrust::transform_reduce(
    thrust::device,
    pts.begin(), pts.end(),
    [=] __device__ (point_t const p) -> size_t
    {
      auto const tmp =
         regulus::loc(vtx[0], vtx[1], vtx[2], vtx[3], p);

      return (tmp < 16 ? 1 : 0);
    },
    size_t{0},
    thrust::plus<size_t>{});

  return tmp == pts.size();
}

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
  auto       d_pts = thrust::device_vector<point_t>{h_pts};
  auto const vtx   =
    regulus::build_root_tetrahedron<point_t>(
      d_pts.begin(),
      d_pts.end());

  // as a precaution, make sure the root tetrahedron we're proposing
  // is positively oriented
  REQUIRE(
    (regulus::orient(
      vtx[0],
      vtx[1],
      vtx[2],
      vtx[3]) == regulus::orientation::positive));

  // and now finally ensure that each point of cartesian grid
  // is contained by the proposed all-encompassing tetrahedron
  auto const all_are_contained = pts_are_contained(vtx, d_pts);

  REQUIRE(all_are_contained);
}