#ifndef REGULUS_ALGORITHM_LOCATION_HPP_
#define REGULUS_ALGORITHM_LOCATION_HPP_

#include <climits>

#include "regulus/loc.hpp"
#include "regulus/type_traits.hpp"
#include "regulus/point_traits.hpp"

#include "regulus/algorithm/orient.hpp"

namespace regulus
{
  template <
    typename Point,
    typename = std::enable_if_t<is_point_v<Point>>
  >
  __host__ __device__
  auto loc(
    Point const a,
    Point const b,
    Point const c,
    Point const d,
    Point const p) -> loc_t
  {
    using coord_type = typename point_traits<Point>::value_type;

    auto loc_code         = loc_t{0};
    auto curr_orientation = loc_t{0};

    // 321
    curr_orientation = static_cast<loc_t>(orient(d, c, b, p));
    if (curr_orientation > 1) {
      return outside_v;
    }

    loc_code |= (curr_orientation << 0);

    // 023
    curr_orientation = static_cast<loc_t>(orient(a, c, d, p));
    if (curr_orientation > 1) {
      return outside_v;
    }

    loc_code |= (curr_orientation << 1);

    // 031
    curr_orientation = static_cast<loc_t>(orient(a, d, b, p));
    if (curr_orientation > 1) {
      return outside_v;
    }

    loc_code |= (curr_orientation << 2);

    // 012
    curr_orientation = static_cast<loc_t>(orient(a, b, c, p));
    if (curr_orientation > 1) {
      return outside_v;
    }

    loc_code |= (curr_orientation << 3);

    return loc_code;
  }
}

#endif // REGULUS_ALGORITHM_LOCATION_HPP_