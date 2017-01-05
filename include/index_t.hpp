#ifndef REGULUS_INDEX_T_HPP_
#define REGULUS_INDEX_T_HPP_

#include "enable_if.hpp"
#include <type_traits>
#include <iostream>

// This struct pattern is shamelessly ripped from Boost's
// strong typedef implementation

template <typename T, typename U>
struct less_than_comparable_base
{
  // equality comparable
  __host__ __device__
  friend auto operator==(U const& y, T const& x) -> bool
  { return x == y; }

  __host__ __device__
  friend auto operator!=(U const& y, T const& x) -> bool
  { return !(x == y); }

  __host__ __device__
  friend auto operator!=(T const& y, U const& x) -> bool
  { return !(y == x); }

  // less than comparable
  __host__ __device__
  friend auto operator<=(T const& x, U const& y) -> bool
  { return !(x > y); }

  __host__ __device__
  friend auto operator>=(T const& x, U const& y) -> bool
  { return !(x < y); }

  __host__ __device__
  friend auto operator>(U const& x, T const& y) -> bool
  { return y < x; }

  __host__ __device__
  friend auto operator<(U const& x, T const& y) -> bool
  { return y > x; }

  __host__ __device__
  friend auto operator<=(U const& x, T const& y) -> bool
  { return !(y < x); }

  __host__ __device__
  friend auto operator>=(U const& x, T const& y) -> bool
  { return !(y > x); }
};

template <typename T, typename D>
struct less_than_comparable : less_than_comparable_base<D, T>
{
  // equality comparible
  __host__ __device__
  friend auto operator!=(T const& x, T const& y) -> bool
  { return !(x == y); }

  // less than comparable
  __host__ __device__
  friend auto operator>(T const& x, T const& y) -> bool
  { return y < x; }

  __host__ __device__
  friend auto operator<=(T const&, T const& y) -> bool
  { return !(y < x); }

  __host__ __device__
  friend auto operator>=(T const& x, T const& y) -> bool
  { return !(x < y); }
};

template <
  typename T,
  typename = enable_if_t<std::is_integral<T>::value>,
  typename = enable_if_t<std::is_signed<T>::value>>
struct maybe_int_t : less_than_comparable<maybe_int_t<T>, T>
{
public:
  using value_type = T;
  using unsigned_value_type = typename std::make_unsigned<T>::type;

  T v;

  // constructors

  // default = invalid value
  __host__ __device__ 
  maybe_int_t(void) : v{-1} {}

  // simple copy and move constructors
  __host__ __device__ 
  maybe_int_t(T u) : v{u} {}

  __host__ __device__ 
  maybe_int_t(maybe_int_t const& other) : v{other.v} {}

  __host__ __device__ 
  maybe_int_t(maybe_int_t&& other) : v{std::move(other.v)} {}

  // implicit conversion operators
  __host__ __device__
  explicit operator bool(void) const 
  { return v >= 0; }

  __host__ __device__
  operator unsigned_value_type(void) const
  { return static_cast<unsigned_value_type>(v); }

  // copy and move assignment operators
  __host__ __device__
  auto operator=(maybe_int_t const& other) -> maybe_int_t&
  { v = other.v; return *this; }

  __host__ __device__
  auto operator=(maybe_int_t&& other) -> maybe_int_t&
  { v = std::move(other.v); return *this; }

  // our base classes can define everything in terms
  // of these two operator overloads
  __host__ __device__
  auto operator==(maybe_int_t const& rhs) const -> bool
  { return v == rhs.v; }

  __host__ __device__
  auto operator<(maybe_int_t const& rhs) const -> bool
  { return v < rhs.v; }
};

template <typename T>
auto operator<<(std::ostream& os, maybe_int_t<T> const& m) -> std::ostream&
{
  os << m.v;
  return os;
}

using index_t = maybe_int_t<ptrdiff_t>;
using loc_t = maybe_int_t<char>;

#endif // REGULUS_INDEX_T_HPP_