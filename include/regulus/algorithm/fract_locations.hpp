#ifndef REGULUS_ALGORITHM_FRACT_LOCATIONS_HPP_
#define REGULUS_ALGORITHM_FRACT_LOCATIONS_HPP_

#include <cstddef>
#include "regulus/views/span.hpp"

namespace regulus
{
  auto fract_locations(
    span<std::ptrdiff_t const> const pa,
    span<loc_t          const> const la,
    span<bool           const> const nm,
    span<std::ptrdiff_t>       const fl) -> void;
}

#endif // REGULUS_ALGORITHM_FRACT_LOCATIONS_HPP_