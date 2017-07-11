#ifndef REGULUS_VECTOR_HPP_
#define REGULUS_VECTOR_HPP_

#include <type_traits>
#include <initializer_list>

#include <thrust/uninitialized_copy.h>
#include <thrust/execution_policy.h>
#include <thrust/inner_product.h>

#include "regulus/array.hpp"

namespace regulus
{

template <
  typename T, size_t L,
  typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
struct vector : public array<T, L>
{
  __host__ __device__
  vector(void) = default;

  __host__ __device__
  vector(std::initializer_list<T> const vals)
  {
    thrust::uninitialized_copy(
      thrust::seq,
      vals.begin(), vals.end(),
      this->data());
  }
};

template <typename T, size_t L>
__host__ __device__
auto operator*(vector<T, L> const x, vector<T, L> const y) 
-> typename vector<T, L>::value_type
{
  return thrust::inner_product(
    thrust::seq,
    x.begin(), x.end(),
    y.begin(),
    typename vector<T, L>::value_type{0});
}

}

#endif // REGULUS_VECTOR_HPP_