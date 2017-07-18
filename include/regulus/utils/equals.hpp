#ifndef REGULUS_UTILS_EQUALS_HPP_
#define REGULUS_UTILS_EQUALS_HPP_

#include <cstdio>
#include "regulus/utils/numeric_limits.hpp"

namespace regulus
{
  template <typename T>
  __host__ __device__
  auto eq(T const x, T const y) -> bool
  {
    auto const eps = numeric_limits<T>::epsilon();
    auto const tmp = fmin(abs(x), abs(y));
    return abs(x - y) <= (tmp * eps);
  }
} // regulus

#endif // REGULUS_UTILS_EQUALS_HPP_