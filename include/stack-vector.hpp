#ifndef REGULUS_STACK_VECTOR_HPP_
#define REGULUS_STACK_VECTOR_HPP_

#include <type_traits>

template <size_t Len, size_t Align>
using aligned_storage_t = typename std::aligned_storage<Len, Align>::type;

template <typename T, int N>
struct stack_vector
{
public:
  using size_type = int;
  using pointer = T*;
  using reference = T&;
  using const_pointer = T const*;
  using const_reference = T const&;
  
private:
  aligned_storage_t<sizeof(T), alignof(T)> data_[N];
  size_type size_ = 0;
  
public:
  __host__ __device__
  stack_vector(void) = default;
  
  __host__ __device__
  stack_vector(const_reference v)
  {
    for (int i = 0; i < N; ++i) {
      emplace_back(v);
    }
  }
  
  __host__ __device__
  ~stack_vector(void)
  {
    for (size_type i = 0; i < size_; ++i) {
      reinterpret_cast<T const*>(data_ + i)->~T();
    }
  }
  
  template <typename ...Args>
  __host__ __device__
  auto emplace_back(Args&& ...args) -> void
  {
    new(data_ + size_) T{std::forward<Args>(args)...};
    ++size_;
  }
  
  __host__ __device__
  auto push_back(const_reference val) -> void
  {
    *reinterpret_cast<pointer>(data_ + size_) = val;
    ++size_;
  }
  
  __host__ __device__
  auto size(void) const -> size_type
  {
    return size_;
  }
  
  __host__ __device__
  auto operator[](size_type const pos) -> reference
  {
    return *reinterpret_cast<pointer>(data_ + pos);
  }
  
  __host__ __device__
  auto operator[](size_type const pos) const -> const_reference
  {
    return *reinterpret_cast<const_pointer>(data_ + pos);
  }
  
  __host__ __device__
  auto begin(void) -> pointer
  {
    return reinterpret_cast<pointer>(data_);
  }
  
  __host__ __device__
  auto begin(void) const -> const_pointer
  {
    return reinterpret_cast<const_pointer>(data_);
  }
  
  __host__ __device__
  auto end(void) -> pointer
  {
    return reinterpret_cast<pointer>(data_ + size_);
  }
  
  __host__ __device__
  auto end(void) const -> const_pointer
  {
    return reinterpret_cast<const_pointer>(data_ + size_);
  }
};

#endif // REGULUS_STACK_VECTOR_HPP_
