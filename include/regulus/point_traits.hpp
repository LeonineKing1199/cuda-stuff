#ifndef REGULUS_POINT_TRAITS_HPP_
#define REGULUS_POINT_TRAITS_HPP_

#include "regulus/type_traits.hpp"

namespace regulus {

  template <
    typename Point,
    typename = typename std::enable_if_t<is_point_v<Point>>
  >
  struct point_traits
  {
    using value_type = typename std::decay_t<decltype(std::declval<Point>().x)>;
  };

}

#endif // REGULUS_POINT_TRAITS_HPP_