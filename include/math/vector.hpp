#ifndef REGULUS_MATH_VECTOR_HPP_
#define REGULUS_MATH_VECTOR_HPP_

#include "array.hpp"
#include "enable_if.hpp"

template <
  typename T, 
  long long L,
  typename = enable_if_t<std::is_floating_point<T>::value>
>
using vector = array<T, L>;

template <
  typename T,
  long long L
>
__host__ __device__
auto operator*(vector<T, L> const& a, vector<T, L> const& b) -> T
{
  T sum = 0;
  for (typename vector<T, L>::size_type i = 0; i < a.size(); ++i) {
   sum += a[i] * b[i];
  }
  return sum;
}

#endif // REGULUS_MATH_VECTOR_HPP_
