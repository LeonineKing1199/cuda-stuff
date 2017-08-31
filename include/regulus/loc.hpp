#ifndef REGULUS_LOC_HPP_
#define REGULUS_LOC_HPP_

#include "regulus/utils/numeric_limits.hpp"

namespace regulus
{
  using loc_t = uint8_t;

  constexpr
  loc_t outside_v = numeric_limits<loc_t>::max();
}

#endif // REGULUS_LOC_HPP_