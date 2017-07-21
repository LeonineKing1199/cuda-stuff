#ifndef REGULUS_UTILS_ROUND_TO_HPP_
#define REGULUS_UTILS_ROUND_TO_HPP_

#include <type_traits>

// This is a small little helper function that'll take a floating point
// value and round it to nearest digit. For example, if we have
// 1.1234567 and we called round_to with a digits value of 3, we'd receive
// 1.123.
// Useful for leveling off floats for "dumber" comparisons
// There are two implementations because CUDA sucks and `powf` is for
// all types except double which is then `pow`. We do this to silence
// narrowing conversion warnings.

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