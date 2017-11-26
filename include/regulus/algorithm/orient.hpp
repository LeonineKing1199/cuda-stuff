#ifndef REGULUS_ALGORITHM_ORIENTATION_HPP_
#define REGULUS_ALGORITHM_ORIENTATION_HPP_

#include "regulus/type_traits.hpp"
#include "regulus/orientation.hpp"
#include "regulus/point_traits.hpp"

#include "regulus/utils/equals.hpp"
#include "regulus/utils/dist_from_plane.hpp"

namespace regulus
{
  template <
    typename Point,
    typename = typename std::enable_if_t<is_point_v<Point>>
  >
  __host__ __device__
  auto orient(
    Point const a,
    Point const b,
    Point const c,
    Point const d) -> orientation
  {
    using coord_type = typename point_traits<Point>::value_type;

    auto const plane_dist = planar_dist(a, b, c, d);

    return (
      eq(plane_dist, coord_type{0})
      ? orientation::zero
      : (
        plane_dist < 0
        ? orientation::negative
        : orientation::positive));
  }
}

#endif // REGULUS_ALGORITHM_ORIENTATION_HPP_