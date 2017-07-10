#ifndef REGULUS_POINT_TRAITS_HPP_
#define REGULUS_POINT_TRAITS_HPP_

#include <type_traits>
#include "regulus/is_point.hpp"

namespace regulus {

template <
  typename Point,
  typename = typename std::enable_if<is_point<Point>::value>::type
>
struct point_traits
{
  using value_type = typename std::decay<decltype(std::declval<Point>().x)>::type;
};

}

#endif // REGULUS_POINT_TRAITS_HPP_