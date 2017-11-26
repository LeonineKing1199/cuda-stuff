#ifndef REGULUS_ALGORITHM_INSPHERE_HPP_
#define REGULUS_ALGORITHM_INSPHERE_HPP_

#include "regulus/matrix.hpp"
#include "regulus/type_traits.hpp"
#include "regulus/orientation.hpp"
#include "regulus/point_traits.hpp"

namespace regulus
{
  template <typename Point>
  __host__ __device__
  auto mag_squared(Point const p)
  -> typename point_traits<Point>::value_type
  {
    return pow(p.x, 2) + pow(p.y, 2) + pow(p.z, 2);
  }

  template <
    typename Point,
    typename = std::enable_if_t<is_point_v<Point>>
  >
  __host__ __device__
  auto insphere(
    Point const a,
    Point const b,
    Point const c,
    Point const d,
    Point const p) -> orientation
  {
    using coord_type = typename point_traits<Point>::value_type;

    auto const det_v = det(
      matrix<coord_type, 5, 5>{
        1, a.x, a.y, a.z, mag_squared(a),
        1, b.x, b.y, b.z, mag_squared(b),
        1, c.x, c.y, c.z, mag_squared(c),
        1, d.x, d.y, d.z, mag_squared(d),
        1, p.x, p.y, p.z, mag_squared(p)});

    return (
      eq(det_v, coord_type{0})
      ? orientation::zero
      : (
        det_v < 0
        ? orientation::negative
        : orientation::positive));
  }
}

#endif // REGULUS_ALGORITHM_INSPHERE_HPP_