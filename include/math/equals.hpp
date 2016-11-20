#ifndef REGULUS_EQUALS_HPP_
#define REGULUS_EQUALS_HPP_

#include <cmath>
#include <limits>
#include <cstdint>

#define absolute(a) ((a) >= 0.0 ? (a) : -(a))

template <typename T>
struct epsilon {};

template <>
struct epsilon<float>
{
  constexpr float static const value = FLT_EPSILON;
};

template <>
struct epsilon<double>
{
  constexpr double static const value = DBL_EPSILON;
};

template <typename T>
__host__ __device__
__inline__
auto eq(T const& a, T const& b) -> bool
{
  return absolute(a - b) <= (
    (absolute(a) > absolute(b) ? absolute(b) : absolute(a)) * epsilon<T>::value);
}





template <typename T>
__host__ __device__
__inline__
auto round_to(T const t, int const digits) -> T
{
  int const factor = pow(10.0, digits);
  T const val = t * factor;
  
  if (val < 0) {
    return ceil(val - 0.5) / factor;
  }
  
  return floor(val + 0.5) / factor;
}

#endif // REGULUS_EQUALS_HPP_
