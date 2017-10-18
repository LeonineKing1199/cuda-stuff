#include "regulus/globals.hpp"
#include "regulus/algorithm/fracture.hpp"

namespace regulus
{
  __global__
  void fracture_kernel(
    span<std::ptrdiff_t const> const pa,
    span<std::ptrdiff_t const> const ta,
    span<loc_t          const> const la,
    span<bool           const> const nm,
    span<std::ptrdiff_t const> const fl,
    span<tetra_t>              const mesh)
  {
    for (auto const tid = get_tid; tid < pa.size(); tid += grid_stride()) {

    }
  }
}