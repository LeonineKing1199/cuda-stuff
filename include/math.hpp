#ifndef REGULUS_MATH_HPP_
#define REGULUS_MATH_HPP_

#include <array>

#include "common.hpp"

// N x M matrix of type T
template <typename T, int N, int M>
using matrix_t = std::array<T, N * M>;

template <
  typename T,
  int N,
  typename = reg::enable_if_t<std::is_floating_point<T>::value>>
auto det(matrix_t<T, N, N>& m) -> T;

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

#endif // REGULUS_MATH_HPP_
