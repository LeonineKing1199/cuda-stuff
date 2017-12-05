#include <iostream>

#include <thrust/device_vector.h>

#include "regulus/loc.hpp"
#include "regulus/tetra.hpp"

#include "regulus/views/span.hpp"

#include "regulus/utils/gen_cartesian_domain.hpp"

#include "regulus/algorithm/nominate.hpp"
#include "regulus/algorithm/fracture.hpp"
#include "regulus/algorithm/assoc_locations.hpp"
#include "regulus/algorithm/fract_locations.hpp"
#include "regulus/algorithm/redistribute_pts.hpp"
#include "regulus/algorithm/make_assoc_relations.hpp"
#include "regulus/algorithm/mark_nominated_tetra.hpp"
#include "regulus/algorithm/build_root_tetrahedron.hpp"
// #include "regulus/algorithm/redistribution_cleanup.hpp"

#include <catch.hpp>

using std::size_t;
using std::ptrdiff_t;

using thrust::host_vector;
using thrust::device_vector;

using regulus::make_span;
using regulus::make_const_span;

TEST_CASE("Point redistribution")
{
  SECTION("should function as expected")
  {
    using point_t = float3;

    // we'll test redistribution in the case of a 9x9x9
    auto const grid_length = size_t{9};
    auto       pts         =
      ([=](void) -> device_vector<point_t>
      {
        auto h_pts = host_vector<point_t>{};
        h_pts.reserve(grid_length * grid_length * grid_length);

        regulus::gen_cartesian_domain<point_t>(
          grid_length,
          std::back_inserter(h_pts));

        return {h_pts};
      })();

    auto const num_pts = pts.size();
    REQUIRE(num_pts == (grid_length * grid_length * grid_length));

    // we're only doing one round of insertion so at a maximum, only 4
    // tetrahedra will exist concurrently
    auto mesh = device_vector<regulus::tetra_t>{4, regulus::tetra_t{-1, -1, -1, -1}};
    REQUIRE(mesh.size() == 4);

    // we use a somewhat greedy estimate for our number of
    // associations
    auto const est_num_assocs = size_t{8 * num_pts};

    auto ta = device_vector<ptrdiff_t>{est_num_assocs, -1};
    auto pa = device_vector<ptrdiff_t>{est_num_assocs, -1};
    auto la =
      device_vector<regulus::loc_t>{est_num_assocs, regulus::outside_v};

    auto fl = device_vector<ptrdiff_t>{est_num_assocs, -1};
    auto nm = device_vector<bool>{num_pts, false};

    // manually nominate a point in the middle
    nm[num_pts / 2] = true;

    point_t p = pts[num_pts / 2];

    // std::cout << "{ " << p.x << ", " << p.y << ", " << p.z << " }" << "\n";

    auto const root_vtx =
      regulus::build_root_tetrahedron<point_t>(pts.begin(), pts.end());

    auto const root_vert_idx =static_cast<ptrdiff_t>(num_pts);
    for (auto const pt : root_vtx) {
      pts.push_back(pt);
    }

    mesh[0] = regulus::tetra_t{
      root_vert_idx,
      root_vert_idx + 1,
      root_vert_idx + 2,
      root_vert_idx + 3};

    auto const num_tetra  = size_t{1};
    auto const assoc_size = num_pts;

    auto nt = device_vector<ptrdiff_t>{num_tetra, -1};
    auto al = device_vector<ptrdiff_t>{assoc_size, -1};

    auto const const_pa_view = make_const_span(pa).subspan(0, assoc_size);
    auto const const_ta_view = make_const_span(ta).subspan(0, assoc_size);
    auto const const_la_view = make_const_span(la).subspan(0, assoc_size);
    auto const const_fl_view = make_const_span(fl).subspan(0, assoc_size);

    auto const const_old_mesh_view = make_const_span(mesh).subspan(0, 1);

    // write pts.size() assocations to { pa, ta, la }
    regulus::make_assoc_relations<point_t>(root_vtx, pts, pa, ta, la);
    regulus::fract_locations(
      const_pa_view,
      const_la_view,
      nm,
      regulus::make_span(fl).subspan(0, num_pts));

    regulus::mark_nominated_tetra(
        const_ta_view,
        const_pa_view,
        nm,
        nt);

    regulus::assoc_locations(const_ta_view, nt, al);
    regulus::fracture(
        num_tetra,
        const_pa_view,
        const_ta_view,
        const_la_view,
        nm,
        const_fl_view,
        mesh);

    regulus::redistribute_pts<point_t>(
      assoc_size,
      ta,
      pa,
      la,
      fl,
      al,
      nt,
      const_old_mesh_view,
      pts);
    cudaDeviceSynchronize();

    // now we need to do some clean-up to get our data ready for processing
    auto zip_begin = thrust::make_zip_iterator(
      thrust::make_tuple(
        pa.begin(), 
        ta.begin(), 
        la.begin()));

    auto zip_end = zip_begin + pa.size();
  }
}