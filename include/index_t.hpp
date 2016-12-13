#ifndef REGULUS_INDEX_T_HPP_
#define REGULUS_INDEX_T_HPP_

#include "enable_if.hpp"
#include <type_traits>

template <
  typename T,
  typename = enable_if_t<std::is_integral<T>::value>,
  typename = enable_if_t<std::is_signed<T>::value>>
struct maybe_int_t
{
public:

  using value_type = T;
  using unsigned_value_type = typename std::make_unsigned<T>::type;

  T v;

  // constructors

  // default = invalid
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

  // equality-based overloads
  __host__ __device__
  auto operator==(maybe_int_t const& other) const -> bool
  { return v == other.v; }

  __host__ __device__
  auto operator!=(maybe_int_t const& other) const -> bool
  { return v != other.v; }

  __host__ __device__
  auto operator<(maybe_int_t const& other) const -> bool
  { return v < other.v; }

  __host__ __device__
  auto operator>(maybe_int_t const& other) const -> bool
  { return v > other.v; }

  __host__ __device__
  auto operator+(T const& t) const -> maybe_int_t
  { return {v + t}; }
};

using index_t = maybe_int_t<long long int>;
using loc_t = maybe_int_t<char>;

#endif // REGULUS_INDEX_T_HPP_