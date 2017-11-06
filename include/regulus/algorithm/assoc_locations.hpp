#ifndef REGULUS_ALGORITHM_ASSOC_LOCATIONS_HPP_
#define REGULUS_ALGORITHM_ASSOC_LOCATIONS_HPP_

#include <cstddef>
#include "regulus/views/span.hpp"

namespace regulus
{
  auto assoc_locations(
    span<std::ptrdiff_t const> const ta,
    span<std::ptrdiff_t const> const nt,
    span<std::ptrdiff_t>       const al) -> void;
}

#endif // REGULUS_ALGORITHM_ASSOC_LOCATIONS_HPP_