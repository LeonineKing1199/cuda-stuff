#ifndef REGULUS_ALGORITHM_NOMINATE_HPP_
#define REGULUS_ALGORITHM_NOMINATE_HPP_

#include "regulus/loc.hpp"
#include "regulus/views/span.hpp"

namespace regulus
{
  auto nominate(
    std::size_t const    assoc_size,
    span<std::ptrdiff_t> pa,
    span<std::ptrdiff_t> ta,
    span<loc_t>          la,
    span<bool>           nm) -> void;
}

#endif // REGULUS_ALGORITHM_NOMINATE_HPP_