#ifndef REGULUS_INDEX_T_HPP_
#define REGULUS_INDEX_T_HPP_

#include "enable_if.hpp"
#include <type_traits>

struct index_t
{
	long long int v;

	__host__ __device__ index_t(void) : v{-1} {}
  __host__ __device__ index_t(long long int u) : v{u} {}
  __host__ __device__ index_t(index_t const& other) : v{other.v} {}
  __host__ __device__ index_t(index_t&& other) : v{other.v} {}

  __host__ __device__
	operator bool(void) const 
  { return v >= 0; }

  __host__ __device__
  operator unsigned long long int(void) const
  {
    return static_cast<unsigned long long int>(v);
  }

  __host__ __device__
  auto operator=(index_t const& other) -> index_t&
  { v = other.v; return *this; }

  __host__ __device__
  auto operator=(index_t&& other) -> index_t&
  { v = other.v; return *this; }

  __host__ __device__
  auto operator==(index_t const& other) const -> bool
  { return v == other.v; }

  __host__ __device__
  auto operator!=(index_t const& other) const -> bool
  { return v != other.v; }

  __host__ __device__
  auto operator<(index_t const& other) const -> bool
  { return v < other.v; }

  __host__ __device__
  auto operator>(index_t const& other) const -> bool
  { return v > other.v; }

  template <
    typename T,
    typename = enable_if_t<std::is_integral<T>::value>
  >
  __host__ __device__
  auto operator+(T const& t) const -> index_t
  {
    return {v + t};
  }
};

#endif // REGULUS_INDEX_T_HPP_