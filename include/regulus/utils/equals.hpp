#ifndef REGULUS_UTILS_EQUALS_HPP_
#define REGULUS_UTILS_EQUALS_HPP_

#include <limits>

namespace regulus
{
  template <typename T>
  __host__ __device__
  auto eq(T const x, T const y) -> bool
  {
    auto const eps = std::numeric_limits<T>::epsilon();
    auto const tmp = abs(x) > abs(y) ? abs(y) : abs(x);
    return abs(x - y) <= tmp * eps;
  }
} // regulus

#endif // REGULUS_UTILS_EQUALS_HPP_