#ifndef REGULUS_ALGORITHM_REDISTRIBUTE_PTS_HPP_
#define REGULUS_ALGORITHM_REDISTRIBUTE_PTS_HPP_

#include <cstddef>

#include "regulus/loc.hpp"
#include "regulus/array.hpp"
#include "regulus/tetra.hpp"
#include "regulus/globals.hpp"
#include "regulus/type_traits.hpp"

#include "regulus/views/span.hpp"

#include "regulus/algorithm/location.hpp"

/**
 * Unfortunately, redistribution logic can be tricky
 * Relevant conditions:
 *
 * assoc_size = size of valid subspan of ta, pa, la (i.e. not -1 values)
 * fl, al have a size of assoc_size
 * ta, pa, la have a .size() larger than assoc_size for the purposes of writing
 * nt has size num_tetra (as it's valid for every valid value in ta)
 * pts has a size large enough to encompass any valid access by pa (i.e max(pa))
 */

namespace regulus
{
  namespace detail
  {
    template <
      typename Point,
      typename = std::enable_if_t<is_point_v<Point>>
    >
    __global__
    void redistribute_pts_kernel(
      std::size_t                const assoc_size,
      span<std::ptrdiff_t>       const ta,
      span<std::ptrdiff_t>       const pa,
      span<loc_t>                const la,
      span<std::ptrdiff_t const> const fl,
      span<std::ptrdiff_t const> const al,
      span<std::ptrdiff_t const> const nt,
      span<tetra_t        const> const mesh,
      span<Point          const> const pts)
    {
      using std::size_t;
      using std::ptrdiff_t;

      // for every tuple in our association arrays...
      for (auto tid = get_tid(); tid < assoc_size; tid += grid_stride()) {

        // load in the {pa, ta} pair
        auto const pa_id = pa[tid];
        auto const ta_id = ta[tid];

        // use our cache to load in the tuple
        // id that's fracturing our tetrahedron
        // tuple_id is an element of [0, assoc_size) || -1
        auto const tuple_id = nt[ta_id];

        // it's possible this tetrahedron isn't being
        // fractured, so bail early
        if (tuple_id < 0) { continue; }

        // if our current tuple id matches the nominated
        // tuple id, there's no work to do here so bail
        // early as well
        if (tid == tuple_id) { continue; }

        // we now want to build up a list of ids to read our
        // tetrahedra from
        auto const read_ids =
          ([=] __device__ (void) -> array<ptrdiff_t, 4>
          {
            auto const fract_loc  =
              mesh.size() +
              (tuple_id == 0 ? 0 : fl[tuple_id - 1]);

            auto const fract_size = __popc(la[tuple_id]);

            auto pos = size_t{0};
            auto ids = array<ptrdiff_t, 4>{-1, -1, -1, -1};

            ids[++pos] = ta_id;
            for (; pos < fract_size; ++pos) {
              ids[++pos] = fract_loc;
              ++fract_loc;
            }
            return ids;
          })();

        auto write_idx = assoc_size + (tid == 0 ? 0 : al[tid - 1]);

        for (auto const idx : read_ids) {
          if (idx < 0) { continue; }

          auto const t = mesh[idx];

          auto const a = pts[t[0]];
          auto const b = pts[t[1]];
          auto const c = pts[t[2]];
          auto const d = pts[t[3]];

          auto const p = pts[pa_id];

          auto const loc_v = loc(a, b, c, d, p);
          if (loc_v == outside_v) { continue; }

          pa[write_idx] = pa_id;
          ta[write_idx] = idx;
          la[write_idx] = loc_v;
        }
      }
    }
  }

  template <
    typename Point,
    typename = std::enable_if_t<is_point_v<Point>>
  >
  auto redistribute_pts(
    std::size_t                const assoc_size,
    span<std::ptrdiff_t>       const ta,
    span<std::ptrdiff_t>       const pa,
    span<loc_t>                const la,
    span<std::ptrdiff_t const> const fl,
    span<std::ptrdiff_t const> const al,
    span<std::ptrdiff_t const> const nt,
    span<tetra_t        const> const mesh,
    span<Point          const> const pts) -> void
  {
    detail::redistribute_pts_kernel<<<bpg, tpb>>>(
      assoc_size, ta, pa, la, fl, al, nt, mesh, pts);
  }
};

#endif // REGULUS_ALGORITHM_REDISTRIBUTE_PTS_HPP_