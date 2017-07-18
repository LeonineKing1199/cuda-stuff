#ifndef REGULUS_MATRIX_HPP_
#define REGULUS_MATRIX_HPP_

#include <iostream>
#include <type_traits>

#include <thrust/equal.h>
#include <thrust/execution_policy.h>

#include "regulus/array.hpp"
#include "regulus/vector.hpp"
#include "regulus/utils/equals.hpp"

namespace regulus
{
  template <
    typename T,
    size_t R,
    size_t C,
    typename = typename std::enable_if<
      std::is_arithmetic<T>::value
    >::type
  >
  struct matrix
  {
    using array_type    = array<T, R * C>;
    using size_type     = typename array_type::size_type;
    using value_type    = typename array_type::value_type;
    using pointer       = typename array_type::pointer;
    using const_pointer = typename array_type::const_pointer;

    array_type data_;

    __host__ __device__
    auto operator[](size_type const row_idx) -> pointer
    {
      return data_.begin() + (row_idx * C);
    }

    __host__ __device__
    auto operator[](size_type const row_idx) const -> const_pointer
    {
      return data_.begin() + (row_idx * C);
    }

    __host__ __device__
    auto operator==(matrix<T, R, C> const& other) const -> bool
    {
      return thrust::equal(
        thrust::seq,
        data_.begin(), data_.end(),
        other.data_.begin(),
        [](T const x, T const y) -> bool
        {
          return eq(x, y);
        });
    }

    __host__ __device__
    auto operator!=(matrix<T, R, C> const& other) const -> bool
    {
      return !(*this == other);
    }

    __host__ __device__
    auto size(void) const -> size_type
    {
      return data_.size();
    }

    __host__ __device__
    auto num_rows(void) const -> size_type
    {
      return R;
    }

    __host__ __device__
    auto num_cols(void) const -> size_type
    {
      return C;
    }

    __host__ __device__
    auto row(size_type const row_idx) const -> vector<T, C>
    {
      vector<T, C> row;

      auto const array_offset = row_idx * C;

      for (size_type i = 0; i < C; ++i) {
        row[i] = data_[array_offset + i];
      }

      return row;
    }

    __host__ __device__
    auto col(size_type const col_idx) const -> vector<T, R>
    {
      vector<T, R> col;

      for (size_type i = 0; i < R; ++i) {
        col[i] = data_[i * C + col_idx];
      }

      return col;
    }
  };

  // matrix multiplication
  // (N x M) x (M x P)) => (N x P)
  template <
    typename T,
    size_t R1, size_t C1,
    size_t C2
  >
  __host__ __device__
  auto operator*(
    regulus::matrix<T, R1, C1> const& a,
    regulus::matrix<T, C1, C2> const& b)
  -> regulus::matrix<T, R1, C2>
  {
    using size_type = typename regulus::matrix<T, 1, 1>::size_type;

    regulus::matrix<T, R1, C2> c;

    for (size_type i = 0; i < a.num_rows(); ++i) {
      for (size_type j = 0; j < b.num_cols(); ++j) {
        c[i][j] = a.row(i) * b.col(j);
      }
    }

    return c;
  }

  // determinant routines!
  template <typename T>
  __host__ __device__
  auto det(regulus::matrix<T, 1, 1> const& m) -> T
  {
    return m.data_[0];
  }

  template <typename T>
  __host__ __device__
  auto det(regulus::matrix<T, 2, 2> const& m) -> T
  {
    auto const& x = m.data_;
    return (x[0] * x[3]) - (x[1] * x[2]);
  }

  template <typename T>
  __host__ __device__
  auto det(regulus::matrix<T, 3, 3> const& m) -> T
  {
    auto const& x = m.data_;
    return (
      x[0] * x[4] * x[8] +
      x[1] * x[5] * x[6] +
      x[2] * x[3] * x[7] -
      x[2] * x[4] * x[6] -
      x[1] * x[3] * x[8] -
      x[0] * x[5] * x[7]);
  }

  template <typename T, size_t N>
  __host__ __device__
  auto det(regulus::matrix<T, N, N> const& m) -> T
  {  
    using size_type = typename matrix<T, 1,  1>::size_type;

    auto const& x = m.data_;
    auto det_v    = T{0};
    
    auto sub_matrix = regulus::matrix<T, (N - 1), (N- 1)>{0};

    // for each column...
    for (size_type col = 0; col < N; ++col) {
      
      // we copy sub-matrices into our recycled buffer
      auto buff_size = size_type{0};
      for (size_type row_idx = 1; row_idx < N; ++row_idx) {
        for (size_type col_idx = 0; col_idx < N; ++col_idx) {

          if (col_idx == col) { continue; }
          
          
          sub_matrix.data_[buff_size] = x[row_idx * N + col_idx];
          ++buff_size;
        }
      }
      
      auto const det_term = x[col] * det(sub_matrix);

      det_v += (col % 2 == 0 ? det_term : -det_term);
    }
    
    return det_v;
  }

  template <typename T, size_t R, size_t C>
  auto operator<<(std::ostream& os, matrix<T, R, C> const& m) -> std::ostream&
  {
    using size_type = typename matrix<T, R, C>::size_type;

    for (size_type i = 0; i < R; ++i) {
      for (size_type j = 0; j < C; ++j) {

        auto const is_last = bool{(i == (R - 1)) && (j == (C - 1))};

        if (is_last) {
          os << m[i][j];
        } else {
          os << m[i][j] << ", ";
        }
      }

      if (i != (R - 1)) { os << '\n'; }
    }

    return os;
  }
}

#endif // REGULUS_MATRIX_HPP_