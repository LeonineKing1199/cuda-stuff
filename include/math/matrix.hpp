#ifndef REGULUS_MATRIX_HPP_
#define REGULUS_MATRIX_HPP_

#include <type_traits>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include "../common.hpp"
#include "../array.hpp"

// we create a forward declaration so that we may create a
// specialization that we also wish to use in the
// implementation
template <
  typename T,
  int N,
  int M,
  typename
>
class matrix;

// we treat vectors as 1 x L matrices
template <typename T, int L>
using vector = matrix<T, 1, L, reg::enable_if_t<std::is_floating_point<T>::value>>;

// our formal matrix definition
template <
  typename T,
  int N,
  int M,
  typename = reg::enable_if_t<std::is_floating_point<T>::value>
>
struct matrix
{ 
  using array_type = typename reg::array<T, N * M>;
  using value_type = typename array_type::value_type;
  using size_type = typename array_type::size_type;
  using reference = value_type&;
  using const_reference = value_type const&;

  reg::array<T, N * M> data;
    
  __host__ __device__
  auto operator==(matrix<T, N, M> const& other) -> bool
  {
    bool not_equal = false;
    auto const& other_data = other.data;
    
    for (size_type i = 0; i < data.size(); ++i) {
      not_equal = not_equal || (data[i] != other_data[i] );
      
      if (not_equal) {
        return false;
      }
    }
      
    return true; 
  }
  
  __host__ __device__
  auto operator!=(matrix<T, N, M> const& other) -> bool
  {
    return !(*this == other);
  }
  
  __host__ __device__
  auto operator[](int const idx) -> T&
  {
    return data[idx];
  }
  
  __host__ __device__
  auto row(int const idx) const -> vector<T, M>
  {
    vector<T, M> r{ T{} };
    
    for (int i = 0; i < M; ++i) {
      r[i] = data[idx * M + i];
    }
    
    return std::move(r);
  }
  
  __host__ __device__
  auto col(int const idx) const -> vector<T, N>
  {
    vector<T, N> c{ T{} };
    
    for (int i = 0; i < N; ++i) {
      c[i] = data[i * M + idx];
    }
    
    return std::move(c);
  }
};


template <typename T, int L>
__host__ __device__
auto operator*(
  vector<T, L> const& a,
  vector<T, L> const& b)
-> T
{
  vector<T, L> c{ T{} };
  
  thrust::transform(
    thrust::seq,
    a.data.begin(), a.data.end(), b.data.begin(),    // we read from this range
    c.data.begin(),                                  // we write to this one
    thrust::multiplies<T>{});                        // we apply this binary functor
  
  return thrust::reduce(thrust::seq, c.data.begin(), c.data.end());
}

template <typename T, int N, int M, int P>
__host__ __device__
auto operator*(
  matrix<T, N, M> const& a,
  matrix<T, M, P> const& b)
-> matrix<T, N, P>
{
  matrix<T, N, P> c{ T{} };
  
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < P; ++j) {
      c[i * P + j] = a.row(i) * b.col(j);
    }
  }
  
  return std::move(c);
}

#endif // REGULUS_MATRIX_HPP_