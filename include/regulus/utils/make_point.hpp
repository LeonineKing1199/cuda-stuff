#ifndef REGULUS_UTILS_MAKE_POINT_HPP_
#define REGULUS_UTILS_MAKE_POINT_HPP_

#include "regulus/point_traits.hpp"

namespace regulus 
{
  template <typename Point>
  __host__ __device__
  auto make_point(
    typename point_traits<Point>::value_type x,
    typename point_traits<Point>::value_type y,
    typename point_traits<Point>::value_type z)
  {
    Point p;
    p.x = x;
    p.y = y;
    p.z = z;

    return p;
  }
}

#endif // REGULUS_UTILS_MAKE_POINT_HPP_