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
    using value_type = std::decay_t<decltype(std::declval<Point>().x)>;
  };


  template <typename Point>
  using point_traits_vt = typename point_traits<Point>::value_type;
}

#endif // REGULUS_POINT_TRAITS_HPP_