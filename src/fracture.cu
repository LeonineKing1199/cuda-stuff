#include <cstddef>
#include <type_traits>

#include "regulus/array.hpp"
#include "regulus/globals.hpp"
#include "regulus/type_traits.hpp"
#include "regulus/algorithm/fracture.hpp"

namespace
{
  constexpr
  __host__ __device__
  auto is_bit_set(
    regulus::loc_t const loc,
    std::size_t    const pos) noexcept -> bool
  {
    return loc & (1 << pos);
  }

  template <typename ...Ts>
  constexpr
  auto make_array(Ts&& ...ts)
  -> regulus::array<std::size_t, sizeof...(ts)>
  {
    return {static_cast<std::size_t>(ts)...};
  }

  // fiv = face-index-vertices
  // i.e. this index in the array contains
  // the relative vertices of a tetraherdon
  // that define a face
  // face 0 = 321, face 1 = 023, etc.
  static __device__
  auto const fiv =
    regulus::array<
      regulus::array<std::size_t, 3>, 4>{
        make_array(3, 2, 1), // 0
        make_array(0, 2, 3), // 1
        make_array(0, 3, 1), // 2
        make_array(0, 1, 2)  // 3
    };
}

namespace regulus
{
  __global__
  void fracture_kernel(
    std::size_t                const num_tetra,
    span<std::ptrdiff_t const> const pa,
    span<std::ptrdiff_t const> const ta,
    span<loc_t          const> const la,
    span<bool           const> const nm,
    span<std::ptrdiff_t const> const fl,
    span<tetra_t>              const mesh)
  {
    for (auto tid = get_tid(); tid < pa.size(); tid += grid_stride()) {
      auto const pa_id        = pa[tid];
      auto const is_nominated = nm[pa_id];

      // in grid-stride loops, continue is the proper way to "return" from
      // a function early
      if (!is_nominated) { continue; }

      auto const num_faces = std::size_t{4};

      auto const ta_id = ta[tid];
      auto const tetra = mesh[ta_id];
      auto const loc   = la[tid];

      auto write_idx = ta_id;

      for (auto i = 0; i < num_faces; ++i) {
        if (!is_bit_set(loc, i)) { continue; }

        auto const a = tetra[fiv[i][0]];
        auto const b = tetra[fiv[i][1]];
        auto const c = tetra[fiv[i][2]];
        auto const d = pa_id;

        auto const tmp = tetra_t{a, b, c, d};

        mesh[write_idx] = tmp;

        if (write_idx == ta_id) {
          auto const offset = (tid == 0 ? 0 : fl[tid -1]);
          write_idx = num_tetra + offset;
        } else {
          ++write_idx;
        }
      }
    }
  }
}