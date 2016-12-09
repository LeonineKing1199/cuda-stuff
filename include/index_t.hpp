#ifndef REGULUS_INDEX_T_HPP_
#define REGULUS_INDEX_T_HPP_

struct index_t
{
	long long int v;

	__host__ __device__ index_t(void) : v{-1} {}
  __host__ __device__ index_t(long long int u) : v{u} {}
  __host__ __device__ index_t(index_t const& other) : v{other.v} {}
  __host__ __device__ index_t(index_t&& other) : v{other.v} {}

  __host__ __device__
	operator bool() const 
  { return v >= 0; }

  __host__ __device__
  operator long long int() const 
  { return v; }


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
};

#endif // REGULUS_INDEX_T_HPP_