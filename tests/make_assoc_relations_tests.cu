#include <iostream>
#include <iterator>

#ifdef _MSC_VER
#pragma warning(push)

// thrust/iterator/iterator_adaptor.h(203)
// : warning C4244: '+=': conversion from '__int64' to 'int', possible loss of data
#pragma warning(disable: 4244)

#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>

#pragma warning(pop)

#elif

#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>

#endif

#include "regulus/point.hpp"

#include "regulus/views/span.hpp"

#include "regulus/utils/numeric_limits.hpp"
#include "regulus/utils/gen_cartesian_domain.hpp"

#include "regulus/algorithm/location.hpp"
#include "regulus/algorithm/make_assoc_relations.hpp"
#include "regulus/algorithm/build_root_tetrahedron.hpp"

#include <catch.hpp>

using std::size_t;
using std::ptrdiff_t;

using regulus::span;
using regulus::loc_t;

using point_t = double3;

auto is_valid_locs(span<loc_t const> const la) -> bool
{
  auto const num_valid = thrust::transform_reduce(
    thrust::device,
    la.begin(), la.end(),
    [] __device__ (loc_t const loc) -> size_t
    {
      return (loc > 0 && loc != regulus::outside_v ? 1 : 0);
    },
    size_t{0},
    thrust::plus<size_t>{});

  return num_valid == la.size();
}

TEST_CASE("Making the initial set of association relations should work")
{
  SECTION("Cartesian set")
  {

    auto const grid_length = std::size_t{9};
    auto const num_pts     = std::size_t{grid_length * grid_length * grid_length};

    auto h_pts = thrust::host_vector<point_t>{};
    h_pts.reserve(num_pts);

    regulus::gen_cartesian_domain<point_t>(
        grid_length,
        std::back_inserter(h_pts));

    REQUIRE(h_pts.size() == num_pts);

    auto d_pts = thrust::device_vector<point_t>{h_pts};

    auto const root_vertices =
      regulus::build_root_tetrahedron<point_t>(d_pts.begin(), d_pts.end());

    auto const num_assocs = 4 * num_pts;

    auto pa = thrust::device_vector<std::ptrdiff_t>{num_assocs, -1};
    auto ta = thrust::device_vector<std::ptrdiff_t>{num_assocs, -1};
    auto la = thrust::device_vector<loc_t>{num_assocs, regulus::numeric_limits<loc_t>::max()};

    using regulus::make_span;
    using regulus::make_const_span;

    regulus::make_assoc_relations<point_t>(
      root_vertices,
      make_const_span(d_pts),
      make_span(pa),
      make_span(ta),
      make_span(la));

    cudaDeviceSynchronize();

    REQUIRE((
      is_valid_locs(
        make_const_span(la).subspan(0, num_pts))));

    REQUIRE((
      thrust::equal(
        ta.begin(), ta.begin() + num_pts,
        thrust::make_constant_iterator(ptrdiff_t{0}))));

    REQUIRE((
      thrust::equal(
        pa.begin(), pa.begin() + num_pts,
        thrust::make_counting_iterator(std::ptrdiff_t{0}),
        thrust::equal_to<std::ptrdiff_t>{})));
  }
}