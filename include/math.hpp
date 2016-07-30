#ifndef REGULUS_MATH_HPP_
#define REGULUS_MATH_HPP_

#include <array>
#include <type_traits>

#include "common.hpp"

namespace reg
{
  template <int N>
  struct greater_than_three
    : std::integral_constant<bool, (N > 3)>
  {};
}

// N x M matrix of type T
template <
  typename T,
  int N,
  int M,
  typename = reg::enable_if_t<std::is_floating_point<T>::value>
>
using matrix_t = std::array<T, N * M>;

// 1x1 implementation
template <typename T>
auto det(matrix_t<T, 1, 1>& m) -> T
{
  return m[0];
}

// 2x2 implementation
template <typename T>
auto det(matrix_t<T, 2, 2>& m) -> T
{
  return (m[0] * m[3] - m[2] * m[1]);
}

// 3x3 implementation
template <typename T>
auto det(matrix_t<T, 3, 3>& m) -> T
{
  return (
          m[0] * m[4] * m[8] +
          m[1] * m[5] * m[6] +
          m[2] * m[3] * m[7] -
          m[2] * m[4] * m[6] -
          m[1] * m[3] * m[8] -
          m[0] * m[5] * m[7]);
}

// we now create a generic specialization that covers all cases
// for matrices larger than 4x4
template <
  typename T,
  int N,
  typename = reg::enable_if_t<reg::greater_than_three<N>::value>
>
auto det(matrix_t<T, N, N>& m) -> T
{
  return 1337;
}

#endif // REGULUS_MATH_HPP_
