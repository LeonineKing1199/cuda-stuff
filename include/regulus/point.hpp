#ifndef REGULUS_POINT_HPP_
#define REGULUS_POINT_HPP_

#include "regulus/type_traits.hpp"

namespace regulus
{
  template <
    typename T,
    typename = typename enable_if_t<
      std::is_arithmetic<T>::value>
  >
  struct point_t
  { T x; T y; T z; };
}

#endif // REGULUS_POINT_HPP_