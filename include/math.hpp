#ifndef REGULUS_MATH_HPP_
#define REGULUS_MATH_HPP_

#include <array>
#include <type_traits>
#include <cmath>
#include <algorithm>
#include <functional>
#include <numeric>

#include "common.hpp"

template <int N>
struct greater_than_three
  : std::integral_constant<bool, (N > 3)>
{};

template <int N>
struct greater_than_zero
  : std::integral_constant<bool, (N > 0)>
{};

// N x M matrix of type T
template <
  typename T, int N, int M,
  typename = enable_if_t<std::is_floating_point<T>::value>
>
using matrix_t = std::array<T, N * M>;

// compare two matrices for equality
template <typename T, int N, int M>
auto operator==(
  matrix_t<T, N, M> const& a,
  matrix_t<T, N, M> const& b)
-> bool
{
  for (int i = 0; i < a.size(); ++i)
    if (a[i] != b[i])
      return false;
     
  return true;
}

// a vector of N elements of type T
template <typename T, int N>
using vector_t = matrix_t<T, 1, N>;

// compute the dot product of two vectors
template <typename T, int N>
auto dot(
  vector_t<T, N> const& a,
  vector_t<T, N> const& b)
-> T
{
  vector_t<T, N> c = { 0 };
  
  std::transform(
    a.begin(), a.end(), b.begin(), // we iterate over a and b
    c.begin(),                     // writing to c
    std::multiplies<T>{});         // c[i] = a[i] * b[i]
    
  return std::accumulate(c.begin(), c.end(), 0); // return reduction
}

// get the i'th row of a matrix as a vector type
template <typename T, int N, int M>
auto row(
  matrix_t<T, N, M> const& a,
  int const row_idx)
-> vector_t<T, M>
{
  vector_t<T, M> row{ 0 };
  
  // for the number of columns...
  for (int j = 0; j < M; ++j)
    // iterate the row at row_idx * num_cols
    row[j] = a[row_idx * M + j];
  
  return row;
}

// get the j'th column of a matrix as a vector type
template <typename T, int N, int M>
auto col(
  matrix_t<T, N, M> const& a,
  int const col_idx)
-> vector_t<T, N>
{
  vector_t<T, N> col{ 0 };
  
  // for the number of rows...
  for (int i = 0; i < N; ++i)
    // iterate the column at col_idx
    col[i] = a[i * M + col_idx];
  
  return col;
}

// (N x M) x (M x P) -> (N x P)
// product of two matrices
template <typename T, int N, int M, int P>
auto matrix_mul(
  matrix_t<T, N, M> const& a,
  matrix_t<T, M, P> const& b)
-> matrix_t<T, N, P>
{
  matrix_t<T, N, P> c = { 0 };
  
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < P; ++j)
      c[i * N + j] = dot<T, M>(row<T, N, M>(a, i), col<T, M, P>(b, j));
  
  return c;
}


// Get the P part of PA = LU
template <typename T, int N>
auto pemutation_matrix(matrix_t<T, N, N> const& a) -> matrix_t<T, N, N>
{
  using mat = matrix_t<T, N, N>;
  
  mat p{ 0 };
  
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      p[i * N + j] = (i == j);
    
  for (int i = 0; i < N; ++i) {
    int max_j = i;
    
    for (int j = i; j < N; ++j)
      if (fabs(a[j * N + i]) > fabs(a[max_j * N + i]))
        max_j = j;
      
    if (max_j != i)
      for (int k = 0; k < N; ++k) {
        auto tmp = p[i * N + k];
        p[i * N + k] = p[max_j * N + k];
        p[max_j * N + k] = tmp;
      }
  }
  
  return p;
}



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
  typename = enable_if_t<greater_than_three<N>::value>
>
auto det(matrix_t<T, N, N>& m) -> T
{
  return 1337;
}

#endif // REGULUS_MATH_HPP_
