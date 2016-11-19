#ifndef REGULUS_MATRIX_HPP_
#define REGULUS_MATRIX_HPP_

#include "point.hpp"
#include "array.hpp"
#include "equals.hpp"

#include <type_traits>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/swap.h>

#include <cmath>
#include <limits>
#include <cstdint>

// we create a forward declaration so that we may create a
// specialization that we also wish to use in the
// implementation
template <
  typename T,
  long long N,
  long long M,
  typename
>
class matrix;

// we treat vectors as 1 x L matrices
template <typename T, long long L>
using vector = matrix<T, 1, L, enable_if_t<std::is_floating_point<T>::value>>;

// our formal matrix definition
template <
  typename T,
  long long N,
  long long M,
  typename = enable_if_t<std::is_floating_point<T>::value>
>
struct matrix
{ 
  __host__ __device__
  auto operator==(matrix const& other) -> bool
  {
    return true;
  }
  using array_type = array<T, N * M>;
  using value_type = typename array_type::value_type;
  using size_type = typename array_type::size_type;
  using reference = value_type&;
  using const_reference = value_type const&;

  array<T, N * M> data;
    
  /*__host__ __device__
  auto operator==(matrix<T, N, M> const& other) const -> bool
  {
    auto const& other_data = other.data;
    
    for (size_type i = 0; i < data.size(); ++i) {
      T const x = data[i];
      T const y = other_data[i];
            
      if (!eq(x, y))
        return false;
    }
      
    return true; 
  }
  
  __host__ __device__
  auto operator!=(matrix<T, N, M> const& other) const -> bool
  {
    return !(*this == other);
  }
  
  __host__ __device__
  auto operator[](int const idx) -> T&
  {
    return data[idx];
  }
  
  __host__ __device__
  auto operator[](int const idx) const -> T const&
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
    
    return r;
  }
  
  __host__ __device__
  auto col(int const idx) const -> vector<T, N>
  {
    vector<T, N> c{ T{} };
    
    for (int i = 0; i < N; ++i) {
      c[i] = data[i * M + idx];
    }
    
    return c;
  }
  
  __host__ __device__
  auto swap_rows(size_type const a_idx, size_type const b_idx) -> matrix&
  {
    auto first_a = data.begin() + a_idx * M;
    auto last_a = first_a + M;
    
    auto first_b = data.begin() + b_idx * M;
    
    thrust::swap_ranges(
      thrust::seq,
      first_a,
      last_a,
      first_b);
    
    return *this;
  }
  
  /*__host__ __device__
  auto det(void) const -> T
  {
    matrix P{ 0 };
    matrix L{ 0 };
    matrix U{ 0 };
    
    int const num_perms = pivot(*this, P);
    LU_decompose(*this, P, L, U);
    
    T det = U[0];
    for (int i = 1; i < N; ++i) {
      det *= U[i * N + i];
    }
    
    return (num_perms % 2 == 0 ? 1 : -1) * det;
  }*/
};

/*
// dot product
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

// matrix multiplication
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
  
  return c;
}

// create identity matrix
template <typename T, int N>
__host__ __device__
auto create_diagonal(void) -> matrix<T, N, N>
{
  matrix<T, N, N> p;
  
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      p[i * N + j] = (i == j);
    }
  }
  
  return p;
}

// interim determinant routine until I get this LU decomp stuff working
template <typename T>
__host__ __device__
auto det(matrix<T, 2, 2> const& m) -> T
{
  return m.data[0] * m.data[3] - m.data[1] * m.data[2];
}

template <typename T>
__host__ __device__
auto det(matrix<T, 3, 3> const& m) -> T
{
  array<T, 9> const& d = m.data;
  return (
    d[0] * d[4] * d[8] +
    d[1] * d[5] * d[6] +
    d[2] * d[3] * d[7] -
    d[2] * d[4] * d[6] -
    d[1] * d[3] * d[8] -
    d[0] * d[5] * d[7]);
}

template <typename T, int N>
__host__ __device__
auto det(matrix<T, N, N> const& m) -> T
{  
  array<T, N * N> const& d = m.data;
  matrix<T, N - 1, N - 1> buff{ 0 };
  T det_value{ 0 };
  
  for (int col = 0; col < N; ++col) {
    
    int buff_size = 0;
    for (int i = 1; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        if (j == col)
          continue;
        
        buff[buff_size] = d[i * N + j];
        ++buff_size;
      }
    }
    
    T const det_term = d[col] * det(buff);
    
    det_value += (col % 2 == 0 ? det_term : -det_term);
  }
  
  return det_value;
}

// return a matrix P such that PA returns a permutation of A
// where every diagonal element is the largest value in its
// respective column of A
// ripped from rosettacode.org on LU decomposition page
template <typename T, int N>
__host__ __device__
auto pivot(matrix<T, N, N> const& A) -> matrix<T, N, N>
{
  matrix<T, N, N> P = create_diagonal<T, N>();
  
  for (int i = 0; i < N; ++i) {
    T max = A[i * N + i];
    int row_idx = i;
    
    for (int j = i; j < N; ++j) {
      if (absolute(A[j * N + i]) > absolute(max)) {
        max = A[j * N + i];
        row_idx = j;
      }
    }
    
    if (row_idx != i) {
      P.swap_rows(i, row_idx);
    }
  }
  
  return P;
}

// takes a matrix A and a reference to a matrix P
// and mutates P into a permutation matrix of A,
// returns number of row swaps
template <typename T, int N>
__host__ __device__
auto pivot(
  matrix<T, N, N> const& A,
  matrix<T, N, N>& P)
-> int
{
  // create an initial diagonal matrix
  P = create_diagonal<T, N>();
  
  int num_swaps = 0;
  
  // for every column in A
  for (int j = 0; j < N; ++j) {
    // get the j-th column
    auto col = A.col(j);
    
    // we start with the j-th row
    int row_idx = j;
    
    // for every row in A after j...
    // (remember, a column's length is number of rows)
    for (int i = j; i < N; ++i) {
      // if the current column's value exceeds our current max
      if (fabs(col[i]) > fabs(col[row_idx])) {
        // reassign the max index
        row_idx = i;
      }
    }
    
    // if the maximum index isn't the location of the initial
    // 1 in the diagonal matrix, swap the rows
    if (row_idx != j) {
      P.swap_rows(row_idx, j);
      ++num_swaps;
    }
  }
  
  return num_swaps;
}

// LU decomposition
// because working with arrays and tuples has always
// felt awkward to me, pass in L and U as mutable references
// to matrices
// even with move semantics, no one wants to deal with std::pair
template <typename T, int N>
__host__ __device__
auto LU_decompose(
  matrix<T, N, N> const& a,
  matrix<T, N, N> const& p,
  matrix<T, N, N>& L,
  matrix<T, N, N>& U)
-> void
{
  U = { 0 };
  L = create_diagonal<T, N>();
  
  auto const ap = p * a;
  
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      T s;
      
      if (j <= i) {
        s = 0;
        for (int k = 0; k < j; ++k) {
          s += L[j * N + k] * U[k * N + i];
        }
        
        U[j * N + i] = ap[j * N + i] - s;
      } 
      
      if (j >= i) {
        s = 0;
        for (int k = 0; k < i; ++k) {
          s += L[j * N + k] * U[k * N + i];
        }
        
        L[j * N + i] = (ap[j * N + i] - s) / U[i * N + i];
      }
    }
  }
}

template <typename T, int N>
__host__ __device__
auto LU_decompose(
  matrix<T, N, N> const& a,
  matrix<T, N, N>& L,
  matrix<T, N, N>& U)
-> void
{
  LU_decompose(a, pivot(a), L, U);
}*/

#endif // REGULUS_MATRIX_HPP_
