#include <iostream>
#include <iterator>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "regulus/matrix.hpp"
#include "regulus/utils/equals.hpp"
#include "regulus/utils/dist_from_plane.hpp"
#include "regulus/utils/gen_cartesian_domain.hpp"
#include "regulus/algorithm/build_root_tetrahedron.hpp"

#include <catch.hpp>

TEST_CASE("Our location routine should be numerically stable")
{
  SECTION("simple integral test as floating point type")
  {
    using point_t = double3;

    auto const a = point_t{-7.999999999999998, 15.999999999999998, -7.999999999999998};
    auto const b = point_t{-7.999999999999998, -7.999999999999998, 15.999999999999998};
    auto const c = point_t{15.999999999999998, -7.999999999999998, -7.999999999999998};
    auto const d = point_t{15.999999999999998, 15.999999999999998, 15.999999999999998};

    auto const p = point_t{0,  8,  8};

    auto const det_v0 = regulus::planar_dist(d, c, b, p);
    auto const det_v1 = regulus::planar_dist(a, c, d, p);
    auto const det_v2 = regulus::planar_dist(a, d, b, p);
    auto const det_v3 = regulus::planar_dist(a, b, c, p);

    REQUIRE((regulus::eq(det_v0, 0.0) || det_v0 > 0));
    REQUIRE((regulus::eq(det_v1, 0.0) || det_v1 > 0));
    REQUIRE(!(regulus::eq(det_v2, 0.0) || det_v2 > 0));
    REQUIRE((regulus::eq(det_v3, 0.0) || det_v3 > 0));
  }

  SECTION("using our fiducial tetrahedron function")
  {
    using point_t = double3;

    auto const grid_length = size_t{9};
    auto const num_pts     = size_t{grid_length * grid_length * grid_length};

    auto h_pts = thrust::host_vector<point_t>{};
    h_pts.reserve(num_pts);

    regulus::gen_cartesian_domain<point_t>(
        grid_length,
        std::back_inserter(h_pts));

    REQUIRE(h_pts.size() == num_pts);

    auto d_pts = thrust::device_vector<point_t>{h_pts};

    auto const rv = regulus::build_root_tetrahedron<point_t>(
        d_pts.begin(),
        d_pts.end());

    auto const p = point_t{0,  8,  8};

    auto const dist0 = regulus::planar_dist(rv[3], rv[2], rv[1], p);
    auto const dist1 = regulus::planar_dist(rv[0], rv[2], rv[3], p);
    auto const dist2 = regulus::planar_dist(rv[0], rv[3], rv[1], p);
    auto const dist3 = regulus::planar_dist(rv[0], rv[1], rv[2], p);

    REQUIRE((regulus::eq(dist0, 0.0) || dist0 > 0));
    REQUIRE((regulus::eq(dist1, 0.0) || dist1 > 0));
    REQUIRE((regulus::eq(dist2, 0.0) || dist2 > 0));
    REQUIRE((regulus::eq(dist3, 0.0) || dist3 > 0));
  }
}