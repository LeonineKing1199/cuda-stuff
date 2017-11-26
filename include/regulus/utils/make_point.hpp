#ifndef REGULUS_UTILS_MAKE_POINT_HPP_
#define REGULUS_UTILS_MAKE_POINT_HPP_

#include "regulus/type_traits.hpp"
#include "regulus/point_traits.hpp"

namespace regulus
{
  template <
    typename Point,
    typename = std::enable_if_t<is_point_v<Point>>
  >
  __host__ __device__
  auto make_point(
    point_traits_vt<Point> const x,
    point_traits_vt<Point> const y,
    point_traits_vt<Point> const z) -> Point
  {
    Point p;
    p.x = x;
    p.y = y;
    p.z = z;

    return p;
  }
}

#endif // REGULUS_UTILS_MAKE_POINT_HPP_