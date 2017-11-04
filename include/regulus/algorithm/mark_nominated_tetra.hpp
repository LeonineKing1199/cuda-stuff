#ifndef REGULUS_ALGORITHM_MARK_NOMINATED_TETRA_HPP_
#define REGULUS_ALGORITHM_MARK_NOMINATED_TETRA_HPP_

#include "regulus/views/span.hpp"

namespace regulus
{
  auto mark_nominated_tetra(
    span<ptrdiff_t const> const ta,
    span<ptrdiff_t const> const pa,
    span<bool      const> const nm,
    span<ptrdiff_t>       const nt) -> void;
}

#endif // REGULUS_ALGORITHM_MARK_NOMINATED_TETRA_HPP_