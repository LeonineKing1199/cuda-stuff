#ifndef REGULUS_IS_POINT_HPP_
#define REGULUS_IS_POINT_HPP_

#include <type_traits>

namespace regulus
{
  template <typename Point>
  struct is_point : std::integral_constant<
    bool,
    std::is_arithmetic<
      typename std::decay<decltype(std::declval<Point>().x)>::type
    >::value
    &&
    std::is_same<
      typename std::decay<decltype(std::declval<Point>().x)>::type,
      typename std::decay<decltype(std::declval<Point>().y)>::type
    >::value
    &&
    std::is_same<
      typename std::decay<decltype(std::declval<Point>().x)>::type,
      typename std::decay<decltype(std::declval<Point>().z)>::type
    >::value
  >
  {};


}

#endif // REGULUS_IS_POINT_HPP_