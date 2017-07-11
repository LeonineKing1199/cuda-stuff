#ifndef REGULUS_ARRAY_HPP_
#define REGULUS_ARRAY_HPP_

#include <iostream>

// We're trying to mimic the STL container here but it's woefully
// incomplete when compared to the "real thing". But it does the
// job most of the time.
namespace regulus {

template <typename T, size_t N>
struct array
{
  using value_type      = T;
  using size_type       = size_t;
  using pointer         = value_type*;
  using const_pointer   = value_type const*;
  using iterator        = pointer;
  using const_iterator  = const_pointer;  
  using reference       = value_type&;
  using const_reference = value_type const&;
 
  T data_[N];

  __host__ __device__
  auto operator[](size_type const idx) -> reference
  {
    return data_[idx];
  }
  
  __host__ __device__
  auto operator[](size_type const idx) const -> const_reference
  {
    return data_[idx];
  }
  
  __host__ __device__
  auto operator==(array<T, N> const& other) const -> bool
  {
    bool v{true};

    for (size_type i = 0; i < N && v; ++i) {
      v = (data_[i] == other.data_[i]);
    }

    return v;
  }
  
  __host__ __device__
  auto operator!=(array<T, N> const& other) const -> bool
  {
    bool v{true};

    for (size_type i = 0; i < N && v; ++i) {
      v = (data_[i] == other.data_[i]);
    }

    return !v;
  }
  
  __host__ __device__
  auto begin(void) -> iterator { return data_; }
  
  __host__ __device__
  auto begin(void) const -> const_iterator { return data_; }
  
  __host__ __device__
  auto end(void) -> iterator { return begin() + N; }
  
  __host__ __device__
  auto end(void) const -> const_iterator { return begin() + N; }
  
  __host__ __device__
  auto size(void) const -> size_type { return N; }
  
  __host__ __device__
  auto front(void) -> reference
  {
    return (*this)[0];
  }
  
  __host__ __device__
  auto front(void) const -> const_reference
  {
    return (*this)[0];
  }
  
  __host__ __device__
  auto back(void) -> reference
  {
    return this->operator[](this->size() - 1);
  }
  
  __host__ __device__
  auto back(void) const -> const_reference
  {
    return this->operator[](this->size() - 1);
  }
  
  __host__ __device__
  auto data(void) -> pointer
  {
    return data_;
  }
  
  __host__ __device__
  auto data(void) const -> const_pointer
  {
    return data_;
  }
}; 
}


template <typename T, size_t N>
auto operator<<(std::ostream& os, regulus::array<T, N> const& a) -> std::ostream& 
{
  os << "{ ";

  for (decltype(N) i = 0; i < N; ++i) {
    if (i == N - 1) {
      os << a.data_[i];
    } else {
      os << a.data_[i] << ", ";
    }
  }

  os << " }";

  return os;
}

#endif // REGULUS_ARRAY_HPP_
