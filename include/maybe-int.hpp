#ifndef REGULUS_MAYBE_INT_HPP_
#define REGULUS_MAYBE_INT_HPP_

#include "enable_if.hpp"
#include <type_traits>
#include <iostream>

// Much of this class design was ripped from Boost's
// implementation of strong typedefs

template <
  typename T,
  typename = enable_if_t<std::is_integral<T>::value>,
  typename = enable_if_t<std::is_signed<T>::value>
>
struct maybe_int
{
  using value_type = T;
  using uvalue_type = typename std::make_unsigned<T>::type;

  value_type t;

  __host__ __device__
  maybe_int(value_type const t_) : t{t_} {}

  __host__ __device__
  maybe_int(void) = default;

  __host__ __device__
  maybe_int(maybe_int const& t_) : t{t_.t} {}
  //maybe_int(maybe_int&& t_) : t{static_cast<value_type&&>(t_.t)} {}

  __host__ __device__
  auto operator=(maybe_int const& rhs) -> maybe_int&
  {
    t = rhs.t;
    return *this;
  }

  __host__ __device__
  auto operator=(value_type const& rhs) -> maybe_int&
  {
    t = rhs;
    return *this;
  }

  __host__ __device__
  operator uvalue_type(void) const
  {
    return static_cast<uvalue_type>(t);
  }

  __host__ __host__
  explicit operator bool(void) const
  {
    return t > 0;
  }

  friend
  __host__ __device__
  auto operator<=(maybe_int const& x, value_type const& y) -> bool
  {
    return x.t <= y;
  }

  friend
  __host__ __device__
  auto operator>=(maybe_int const& x, value_type const& y) -> bool
  {
    return x.t >= y;
  }

  friend
  __host__ __device__
  auto operator>(value_type const& y, maybe_int const& x) -> bool
  {
    return y > x.t;
  }

  friend
  __host__ __device__
  auto operator<(value_type const& y, maybe_int const& x) -> bool
  {
    return y < x.t;
  }

  friend
  __host__ __device__
  auto operator<=(value_type const& y, maybe_int const& x) -> bool
  {
    return y <= x.t;
  }

  friend
  __host__ __device__
  auto operator>=(value_type const& y, maybe_int const& x) -> bool
  {
    return y >= x.t;
  }

  friend
  __host__ __device__
  auto operator==(value_type const& y, maybe_int const& x) -> bool
  {
    return y == x.t;
  }

  friend
  __host__ __device__
  auto operator!=(value_type const& y, maybe_int const& x) -> bool
  {
    return y != x.t;
  }

  friend
  __host__ __device__
  auto operator!=(maybe_int const& x, value_type const& y) -> bool
  {
    return x.t != y;
  }

  friend
  __host__ __device__
  auto operator>(maybe_int const& x, maybe_int const& y) -> bool
  {
    return x.t > y.t;
  }

  friend
  __host__ __device__
  auto operator<=(maybe_int const& x, maybe_int const& y) -> bool
  {
    return x.t <= y.t;
  }

  friend
  __host__ __device__
  auto operator>=(maybe_int const& x, maybe_int const& y) -> bool
  {
    return x.t >= y.t;
  }

  friend
  __host__ __device__
  auto operator!=(maybe_int const& x, maybe_int const& y) -> bool
  {
    return x.t != y.t;
  }

  friend
  __host__ __device__
  auto operator==(maybe_int const& x, maybe_int const& y) -> bool
  {
    return x.t == y.t;
  }

  friend
  __host__ __device__
  auto operator<(maybe_int const& x, maybe_int const& y) -> bool
  {
    return x.t < y.t;
  }
};

#endif // REGULUS_MAYBE_INT_HPP_