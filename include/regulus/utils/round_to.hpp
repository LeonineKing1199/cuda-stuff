#ifndef REGULUS_UTILS_ROUND_TO_HPP_
#define REGULUS_UTILS_ROUND_TO_HPP_

#include <type_traits>

namespace regulus
{
  template <typename T>
  __host__ __device__
  auto round_to(T const t, int const digits) -> typename std::enable_if<
      std::is_arithmetic<T>::value &&
      !std::is_same<T, double>::value,
      T
    >::type
  {
    auto const factor = powf(10.0, static_cast<T>(digits));
    auto const val    = t * factor;
    
    if (val < 0) {
      return ceil(val - 0.5) / factor;
    }
    
    return floor(val + 0.5) / factor;
  }

  template <
    typename T
  >
  __host__ __device__
  auto round_to(T const t, int const digits) -> typename std::enable_if<
      std::is_same<T, double>::value,
      T
    >::type
  {
    auto const factor = pow(10.0, static_cast<T>(digits));
    auto const val    = t * factor;
    
    if (val < 0) {
      return ceil(val - 0.5) / factor;
    }
    
    return floor(val + 0.5) / factor;
  }
}

#endif // REGULUS_UTILS_ROUND_TO_HPP_