#ifndef REGULUS_ALGORITHM_REDISTRIBUTION_CLEANUP_HPP_
#define REGULUS_ALGORITHM_REDISTRIBUTION_CLEANUP_HPP_

#include <cstddef>
#include "regulus/loc.hpp"
#include "regulus/views/span.hpp"

namespace regulus
{
  auto redistribution_cleanup(
    span<std::ptrdiff_t> const pa,
    span<std::ptrdiff_t> const ta,
    span<regulus::loc_t> const la,
    span<bool const>     const nm) -> std::ptrdiff_t;
}

#endif // REGULUS_ALGORITHM_REDISTRIBUTION_CLEANUP_HPP_