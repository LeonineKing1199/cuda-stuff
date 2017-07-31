#ifndef REGULUS_UTILS_DIST_FROM_PLANE_HPP_
#define REGULUS_UTILS_DIST_FROM_PLANE_HPP_

#include "regulus/matrix.hpp"
#include "regulus/is_point.hpp"
#include "regulus/type_traits.hpp"
#include "regulus/point_traits.hpp"

namespace regulus
{
  // Function that gets the planar distance of the
  // point d from the plane spanned by abc
  template <
    typename Point,
    typename = enable_if_t< is_point<Point>::value >
  >
  __host__ __device__
  auto planar_dist(
    Point const a,
    Point const b,
    Point const c,
    Point const d)
  -> typename point_traits<Point>::value_type
  {
    using coord_type = typename point_traits<Point>::value_type;

    return det(
      matrix<coord_type, 4, 4>{
        1, a.x, a.y, a.z,
        1, b.x, b.y, b.z,
        1, c.x, c.y, c.z,
        1, d.x, d.y, d.z});
  }
}

#endif // REGULUS_UTILS_DIST_FROM_PLANE_HPP_