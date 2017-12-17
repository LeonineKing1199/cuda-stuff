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
#include "regulus/algorithm/redistribution_cleanup.hpp"

#include <catch.hpp>

using std::size_t;
using std::ptrdiff_t;

using thrust::host_vector;
using thrust::device_vector;

using namespace regulus;

TEST_CASE("Point redistribution")
{
  SECTION("should function as expected")
  {
    using point_t = float3;

    // we'll test redistribution in the case of a 9x9x9
    auto const grid_length = size_t{9};

    // pts will serve as our host-side point buffer
    // it initially contains the Cartesian point set
    auto pts =
      ([=](void) -> device_vector<point_t>
      {
        auto h_pts = host_vector<point_t>{};
        h_pts.reserve(grid_length * grid_length * grid_length);

        gen_cartesian_domain<point_t>(
          grid_length,
          std::back_inserter(h_pts));

        return {h_pts};
      })();

    auto const num_pts = pts.size();
    REQUIRE(num_pts == (grid_length * grid_length * grid_length));

    // we're only doing one round of insertion so at a maximum, only 4
    // tetrahedra will exist concurrently
    auto mesh = device_vector<tetra_t>{4, tetra_t{-1, -1, -1, -1}};
    REQUIRE(mesh.size() == 4);

    // we use a somewhat greedy estimate for our number of
    // associations
    auto const est_num_assocs = size_t{8 * num_pts};

    auto ta = device_vector<ptrdiff_t>{est_num_assocs, -1};
    auto pa = device_vector<ptrdiff_t>{est_num_assocs, -1};
    auto la = device_vector<loc_t>{est_num_assocs, outside_v};

    auto fl = device_vector<ptrdiff_t>{est_num_assocs, -1};
    auto nm = device_vector<bool>{num_pts, false};

    // manually nominate a point in the middle
    nm[num_pts / 2] = true;

    // point_t p = pts[num_pts / 2];
    // std::cout << "{ " << p.x << ", " << p.y << ", " << p.z << " }" << "\n";

    auto const root_vtx =
      build_root_tetrahedron<point_t>(pts.begin(), pts.end());

    // write the all-encompassing root tetrahedron to the mesh buffer
    // the vertices are written to the back of the pts buffer so we
    // know the first vertex begins at the spot `num_pts`
    auto const root_vert_idx = static_cast<ptrdiff_t>(num_pts);
    for (auto const pt : root_vtx) {
      pts.push_back(pt);
    }

    mesh[0] = tetra_t{
      root_vert_idx,
      root_vert_idx + 1,
      root_vert_idx + 2,
      root_vert_idx + 3};

    auto const num_tetra  = size_t{1};
    auto const assoc_size = num_pts;

    auto nt = device_vector<ptrdiff_t>{num_tetra, -1};
    auto al = device_vector<ptrdiff_t>{assoc_size, -1};

    auto const span_gen =
      [](auto const& container, size_t const len)
      {
        return make_const_span(container).subspan(0, len);
      };

    auto const pa_const_view = span_gen(pa, assoc_size);
    auto const ta_const_view = span_gen(ta, assoc_size);
    auto const la_const_view = span_gen(la, assoc_size);
    auto const fl_const_view = span_gen(fl, assoc_size);

    auto const fl_view = make_span(fl).subspan(0, assoc_size);

    auto const const_old_mesh_view = make_const_span(mesh).subspan(0, 1);

    // write `num_pts` assocations to { pa, ta, la }
    // because nm, nt and al are exactly sized, they don't need
    // any form of span slicing
    make_assoc_relations<point_t>(
      root_vtx,
      span_gen(pts, num_pts),
      pa, ta, la);

    fract_locations(
      pa_const_view,
      la_const_view,
      nm,
      fl_view);

    mark_nominated_tetra(
        ta_const_view,
        pa_const_view,
        nm,
        nt);

    assoc_locations(ta_const_view, nt, al);
    fracture(
        num_tetra,
        pa_const_view,
        ta_const_view,
        la_const_view,
        nm,
        fl_const_view,
        mesh);

    redistribute_pts<point_t>(
      assoc_size,
      ta,
      pa,
      la,
      fl,
      al,
      nt,
      const_old_mesh_view,
      pts);

    redistribution_cleanup(pa, ta, la, nm);

    cudaDeviceSynchronize();
  }
}