#ifndef REGULUS_POINT_HPP_
#define REGULUS_POINT_HPP_

#include "regulus/type_traits.hpp"

namespace regulus
{
  template <
    typename T,
    typename = typename std::enable_if_t<is_arithmetic_v<T>>
  >
  struct point_t
  { T x; T y; T z; };
}

#endif // REGULUS_POINT_HPP_