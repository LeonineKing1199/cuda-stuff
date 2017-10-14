#ifndef REGULUS_ALGORITHM_NOMINATE_HPP_
#define REGULUS_ALGORITHM_NOMINATE_HPP_

#include "regulus/loc.hpp"
#include "regulus/views/span.hpp"

namespace regulus
{
  auto nominate(
    span<std::ptrdiff_t> const pa,
    span<std::ptrdiff_t> const ta,
    span<loc_t>          const la,
    span<bool>           const nm) -> void;
}

#endif // REGULUS_ALGORITHM_NOMINATE_HPP_