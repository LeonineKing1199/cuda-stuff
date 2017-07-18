#ifndef REGULUS_ALGORITHM_ORIENTATION_HPP_
#define REGULUS_ALGORITHM_ORIENTATION_HPP_

#include "regulus/is_point.hpp"
#include "regulus/type_traits.hpp"
#include "regulus/matrix.hpp"
#include "regulus/utils/equals.hpp"
#include "regulus/point_traits.hpp"

namespace regulus
{
  enum class orientation { positive, zero, negative };
  
  template <
    typename Point,
    typename = typename enable_if_t<is_point<Point>::value>>
  __host__ __device__
  auto orient(
    Point const a,
    Point const b,
    Point const c,
    Point const d) 
  -> orientation
  {
    auto const det_v = det(
      matrix<typename point_traits<Point>::value_type, 4, 4>{
        1, a.x, a.y, a.z,
        1, b.x, b.y, b.z,
        1, c.x, c.y, c.z,
        1, d.x, d.y, d.z});

    return (
      eq(det_v, 0)
      ? orientation::zero
      : (
        det_v < 0
        ? orientation::negative
        : orientation::positive));
  }
}

#endif // REGULUS_ALGORITHM_ORIENTATION_HPP_