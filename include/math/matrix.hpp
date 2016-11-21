#ifndef REGULUS_MATRIX_HPP_
#define REGULUS_MATRIX_HPP_

#include "point.hpp"
#include "vector.hpp"
#include "array.hpp"
#include "equals.hpp"
#include "enable_if.hpp"


// our formal matrix definition
template <
  typename T,
  long long N, // rows
  long long M, // cols
  typename = enable_if_t<std::is_floating_point<T>::value>
>
struct matrix
{ 
  using array_type = array<T, N * M>;
  using value_type = typename array_type::value_type;
  using size_type = typename array_type::size_type;
  using reference = typename array_type::reference;
  using const_reference = typename array_type::const_reference;
  using pointer = typename array_type::pointer;
  using const_pointer = typename array_type::const_pointer;

  array<T, N * M> data_;
    
  // by using this style of access pattern, we get the 
  // really nice [i][j] syntax
  __host__ __device__  
  auto operator[](size_type const row_idx) -> pointer
  {
    return data_.data() + (row_idx * M);
  }
  
  __host__ __device__
  auto operator[](size_type const row_idx) const -> const_pointer
  {
    return data_.data() + (row_idx * M);
  }
    
  __host__ __device__
  auto operator==(matrix<T, N, M> const& other) const -> bool
  {
    return data_ == other.data_;
  }
  
  __host__ __device__
  auto operator!=(matrix<T, N, M> const& other) const -> bool
  {
    return data_ != other.data_;
  }
  
  __host__ __device__
  auto size(void) const -> size_type
  {
    return data_.size();
  }
  
  // each row has num_cols elements
  __host__ __device__
  auto row(size_type const idx) const -> vector<T, M>
  {
    vector<T, M> r;
    
    for (size_type i = 0; i < M; ++i) {
      r[i] = data_[idx * M + i];
    }
    
    return r;
  }
  
  // each column has num_rows elements
  __host__ __device__
  auto col(size_type const idx) const -> vector<T, N>
  {
    vector<T, N> c;
    
    for (int i = 0; i < N; ++i) {
      c[i] = data_[i * M + idx];
    }
    
    return c;
  }
};


// matrix multiplication
// (N x M) x (M x P)) => (N x P)
template <
  typename T, 
  long long N, 
  long long M, 
  long long P
>
__host__ __device__
auto operator*(
  matrix<T, N, M> const& a,
  matrix<T, M, P> const& b)
-> matrix<T, N, P>
{
  using size_type = typename matrix<T, N, P>::size_type;
  matrix<T, N, P> c;
  for (size_type i = 0; i < N; ++i) {
    for (size_type j = 0; j < P; ++j) {
      c[i][j] = a.row(i) * b.col(j);
    }
  }
  return c;
}

// interim determinant routine until I get LU decomp stuff working
template <typename T>
__host__ __device__
auto det(matrix<T, 1, 1> const& m) -> T
{
  return m.data_[0];
}

template <typename T>
__host__ __device__
auto det(matrix<T, 2, 2> const& m) -> T
{
  return m.data_[0] * m.data_[3] - m.data_[1] * m.data_[2];
}

template <typename T>
__host__ __device__
auto det(matrix<T, 3, 3> const& m) -> T
{
  array<T, 9> const& d = m.data_;
  return (
    d[0] * d[4] * d[8] +
    d[1] * d[5] * d[6] +
    d[2] * d[3] * d[7] -
    d[2] * d[4] * d[6] -
    d[1] * d[3] * d[8] -
    d[0] * d[5] * d[7]);
}

template <typename T, long long N>
__host__ __device__
auto det(matrix<T, N, N> const& m) -> T
{  
  using size_type = typename matrix<T, N,  N>::size_type;

  array<T, N * N> const& d = m.data_;
  matrix<T, N - 1, N - 1> buff{ 0 };
  T det_value{ 0 };
  
  for (size_type col = 0; col < N; ++col) {
    
    size_type buff_size = 0;
    for (size_type i = 1; i < N; ++i) {
      for (size_type j = 0; j < N; ++j) {
        if (j == col)
          continue;
        
        buff.data_[buff_size] = d[i * N + j];
        ++buff_size;
      }
    }
    
    T const det_term = d[col] * det(buff);
    
    det_value += (col % 2 == 0 ? det_term : -det_term);
  }
  
  return det_value;
}

#endif // REGULUS_MATRIX_HPP_
