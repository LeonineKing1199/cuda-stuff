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

  __host__ __device__
  explicit operator bool(void) const
  {
    return t >= 0;
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

template <typename T>
auto operator<<(std::ostream& os, maybe_int<T> const& mi) -> std::ostream&
{
  if (static_cast<bool>(mi)) {
    // it seems weird to chain casts like this...
    // we know if our boolean check passes, it's okay
    // to promote the encapsulated type to unsigned
    // because unsigned chars (common alias for uint8_t),
    // we promote to the largest possible integral type
    // so the value is guaranteed to be contained
    os << static_cast<unsigned long long>(static_cast<typename maybe_int<T>::uvalue_type>(mi));
  } else {
    os << "invalid value";
  }
  return os;
}

#endif // REGULUS_MAYBE_INT_HPP_