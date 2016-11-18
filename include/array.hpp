#ifndef REGULUS_ARRAY_HPP_
#define REGULUS_ARRAY_HPP_

template <typename T, long long N>
struct array
{
  using value_type = T;
  using size_type = long long;
  
  using pointer = value_type*;
  using const_pointer = value_type const*;
  
  using iterator = pointer;
  using const_iterator = const_pointer;  
  
  using reference = value_type&;
  using const_reference = value_type const&;
 
  T data[N];

  __host__ __device__
  auto operator[](size_type const idx) -> reference
  {
    return data[idx];
  }
  
  __host__ __device__
  auto operator[](size_type const idx) const -> const_reference
  {
    return data[idx];
  }
  
  __host__ __device__
  auto operator==(array<T, N> const& other) const -> bool
  {
    bool v{ true };

    for (size_type i = 0; i < N && v; ++i) {
      v = (data[i] == other.data[i]);
    }

    return v;
  }
  
  __host__ __device__
  auto operator!=(array<T, N> const& other) const -> bool
  {
    bool v{ true };

    for (size_type i = 0; i < N && v; ++i) {
      v = (data[i] == other.data[i]);
    }

    return !v;
  }
  
  __host__ __device__
  auto begin(void) -> iterator { return data; }
  
  __host__ __device__
  auto begin(void) const -> const_iterator { return data; }
  
  __host__ __device__
  auto end(void) -> iterator { return begin() + N; }
  
  __host__ __device__
  auto end(void) const -> const_iterator { return begin() + N; }
  
  __host__ __device__
  auto size(void) const -> size_type { return N; }
};  


#endif // REGULUS_ARRAY_HPP_
