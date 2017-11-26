#include <iterator>

#include <thrust/device_vector.h>

#include "regulus/loc.hpp"
#include "regulus/array.hpp"

#include "regulus/utils/gen_cartesian_domain.hpp"

#include "regulus/algorithm/location.hpp"
#include "regulus/algorithm/redistribute_pts.hpp"

#include <catch.hpp>

using std::size_t;
using std::ptrdiff_t;

TEST_CASE("Point redistribution")
{
  SECTION("should function as expected")
  {
    using point_t = float3;

    auto const grid_length = size_t{9};

    auto h_pts = thrust::host_vector<point_t>{};
    h_pts.reserve(grid_length * grid_length * grid_length);

    regulus::gen_cartesian_domain<point_t>(
      grid_length,
      std::back_inserter(h_pts));


  }
}