#ifndef REGULUS_UTILS_MAKE_POINT_HPP_
#define REGULUS_UTILS_MAKE_POINT_HPP_

#include "regulus/is_point.hpp"
#include "regulus/type_traits.hpp"
#include "regulus/point_traits.hpp"

namespace regulus
{
  template <
    typename Point,
    typename = enable_if_t<is_point_v<Point>>,
    typename Coord = typename point_traits<Point>::value_type
  >
  __host__ __device__
  auto make_point(
    Coord const x,
    Coord const y,
    Coord const z) -> Point
  {
    Point p;
    p.x = x;
    p.y = y;
    p.z = z;

    return p;
  }
}

#endif // REGULUS_UTILS_MAKE_POINT_HPP_