#ifndef REGULUS_ALGORITHM_LOCATION_HPP_
#define REGULUS_ALGORITHM_LOCATION_HPP_

#include <climits>

#include "regulus/is_point.hpp"
#include "regulus/type_traits.hpp"
#include "regulus/point_traits.hpp"
#include "regulus/algorithm/orient.hpp"

namespace regulus
{
  template <
    typename Point,
    typename = enable_if_t<is_point<Point>::value>
  >
  __host__ __device__
  auto loc(
    Point const a,
    Point const b,
    Point const c,
    Point const d,
    Point const p)
  -> uint8_t
  {
    using coord_type = typename point_traits<Point>::value_type;

    auto loc_code         = uint8_t{0};
    auto curr_orientation = uint8_t{0};

    // 321
    curr_orientation = static_cast<uint8_t>(orient(d, c, b, p));
    if (curr_orientation > 1) {
      return UINT8_MAX;
    }

    loc_code |= (curr_orientation << 0);

    // 023
    curr_orientation = static_cast<uint8_t>(orient(a, c, d, p));
    if (curr_orientation > 1) {
      return UINT8_MAX;
    }

    loc_code |= (curr_orientation << 1);

    // 031
    curr_orientation = static_cast<uint8_t>(orient(a, d, b, p));
    if (curr_orientation > 1) {
      return UINT8_MAX;
    }

    loc_code |= (curr_orientation << 2);

    // 012
    curr_orientation = static_cast<uint8_t>(orient(a, b, c, p));
    if (curr_orientation > 1) {
      return UINT8_MAX;
    }

    loc_code |= (curr_orientation << 3);

    return loc_code;
  }
}

#endif // REGULUS_ALGORITHM_LOCATION_HPP_