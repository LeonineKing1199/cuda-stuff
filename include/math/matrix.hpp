#ifndef REGULUS_MATRIX_HPP_
#define REGULUS_MATRIX_HPP_

#include <type_traits>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/swap.h>

#include <cmath>

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
  auto operator==(matrix<T, N, M> const& other) const -> bool
  {
    bool not_equal = false;
    auto const& other_data = other.data;
    
    for (size_type i = 0; i < data.size(); ++i) {
      // TODO: somehow replace 1e-5 with something more accurate
      not_equal = not_equal || !(fabs(data[i] - other_data[i]) < 1e-5);
      
      if (not_equal) {
        return false;
      }
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
  
  __host__ __device__
  auto swap_rows(size_type const a_idx, size_type const b_idx) -> void
  {
    auto first_a = data.begin() + a_idx * M;
    auto last_a = first_a + M;
    
    auto first_b = data.begin() + b_idx * M;
    
    thrust::swap_ranges(
      thrust::seq,
      first_a,
      last_a,
      first_b);
  }
};


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
  
  return std::move(c);
}

// create initial triangular matrix
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
  
  return std::move(p);
}

// return a matrix P such that PA returns a permutation of A
// where every diagonal element is the largest value in its
// respective column of A
template <typename T, int N>
__host__ __device__
auto pivot(matrix<T, N, N> const& A) -> matrix<T, N, N>
{
  // create an initial diagonal matrix
  matrix<T, N, N> P = create_diagonal<T, N>();
  
  // for every column in A
  for (int j = 0; j < N; ++j) {
    // get the j-th column
    auto col = A.col(j);
    
    // we start with the j-th row
    int row_idx = j;
    
    // for every row in A after j...
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
    }
  }
  
  // forward P to the caller
  return std::move(P);
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
  matrix<T, N, N>& L,
  matrix<T, N, N>& U)
-> void
{
  U = { 0 };
  L = create_diagonal<T, N>();
  
  auto const ap = pivot(a) * a;
  
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

#endif // REGULUS_MATRIX_HPP_