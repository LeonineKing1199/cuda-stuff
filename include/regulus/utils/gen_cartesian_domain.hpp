#ifndef REGULUS_UTILS_GEN_CARTESIAN_DOMAIN_HPP_
#define REGULUS_UTILS_GEN_CARTESIAN_DOMAIN_HPP_

#include <type_traits>
#include <thrust/host_vector.h>

#include "regulus/is_point.hpp"

namespace regulus {

template <typename Point, typename OutputIterator>
auto gen_cartesian_domain(
  size_t const grid_length, 
  OutputIterator output) -> typename std::enable_if<is_point<Point>::value>::type
{
  using value_type = typename std::decay<decltype(std::declval<Point>().x)>::type;

  for     (size_t i = 0; i < grid_length; ++i) {
    for   (size_t j = 0; j < grid_length; ++j) {
      for (size_t k = 0; k < grid_length; ++k) {
        Point p;
        p.x = static_cast<value_type>(i);
        p.y = static_cast<value_type>(j);
        p.z = static_cast<value_type>(k);

        *output++ = p;
      }
    }
  }
}

}

#endif // REGULUS_UTILS_GEN_CARTESIAN_DOMAIN_HPP_