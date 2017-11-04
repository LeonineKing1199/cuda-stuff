#include <cstddef>

#include "regulus/globals.hpp"
#include "regulus/views/span.hpp"
#include "regulus/algorithm/mark_nominated_tetra.hpp"

namespace
{
  using std::ptrdiff_t;
  using regulus::span;
  using regulus::get_tid;
  using regulus::grid_stride;

  __global__
  void mark_nominated_tetra_kernel(
    span<ptrdiff_t const> const ta,
    span<ptrdiff_t const> const pa,
    span<bool      const> const nm,
    span<ptrdiff_t>       const nt)
  {
    for (auto tid = get_tid(); tid < ta.size(); tid += grid_stride()) {
      auto const ta_id = ta[tid];
      auto const pa_id = pa[tid];

      auto const is_nominated = nm[pa_id];

      atomicMax(
        nt.data() + ta_id,
        is_nominated
        ? static_cast<ptrdiff_t>(tid)
        : -1);
    }
  }
}

namespace regulus
{
  auto mark_nominated_tetra(
    span<ptrdiff_t const> const ta,
    span<ptrdiff_t const> const pa,
    span<bool      const> const nm,
    span<ptrdiff_t>       const nt) -> void
  {
    mark_nominated_tetra_kernel<<<bpg, tpb>>>(ta, pa, nm, nt);
  }
}