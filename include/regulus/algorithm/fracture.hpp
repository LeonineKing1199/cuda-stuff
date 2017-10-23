#ifndef REGULUS_ALGORITHM_FRACTURE_HPP_
#define REGULUS_ALGORITHM_FRACTURE_HPP_

#include "regulus/loc.hpp"
#include "regulus/tetra.hpp"
#include "regulus/views/span.hpp"

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
    span<tetra_t>              const mesh);
}

#endif // REGULUS_ALGORITHM_FRACTURE_HPP_