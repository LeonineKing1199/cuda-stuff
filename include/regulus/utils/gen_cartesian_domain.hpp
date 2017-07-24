#ifndef REGULUS_UTILS_GEN_CARTESIAN_DOMAIN_HPP_
#define REGULUS_UTILS_GEN_CARTESIAN_DOMAIN_HPP_

#include "regulus/point_traits.hpp"
#include "regulus/type_traits.hpp"
#include "regulus/is_point.hpp"
#include "regulus/utils/make_point.hpp"

namespace regulus
{
  template <
    typename Point,
    typename OutputIterator,
    typename = enable_if_t<is_point<Point>::value>
  >
  auto gen_cartesian_domain(
    size_t const grid_length,
    OutputIterator output)
  -> void
  {
    using coord_type = typename point_traits<Point>::value_type;

    for (size_t i = 0; i < grid_length; ++i) {
      for (size_t j = 0; j < grid_length; ++j) {
        for (size_t k = 0; k < grid_length; ++k) {
          *output++ = make_point<Point>(
            static_cast<coord_type>(i),
            static_cast<coord_type>(j),
            static_cast<coord_type>(k));
        }
      }
    }
  }
} // regulus

#endif // REGULUS_UTILS_GEN_CARTESIAN_DOMAIN_HPP_