#ifndef REGULUS_VIEWS_SPAN_HPP_
#define REGULUS_VIEWS_SPAN_HPP_

#include <cstddef>

namespace regulus
{
  template <typename T>
  struct span
  {
  public:
    using value_type = T;
    using pointer    = value_type*;
    using reference  = value_type&;
    using size_type  = std::size_t;

  private:
    pointer   data_;
    size_type size_;

  public:

    // default constructor
    constexpr __host__ __device__
    span(void) noexcept
    : data_{nullptr}, size_{0}
    {}

    // pointer + size_type constructor
    constexpr __host__ __device__
    span(pointer const p, size_type const s)
    : data_{p}, size_{s}
    {}

    // pointer range constructor
    constexpr __host__ __device__
    span(pointer const first, pointer const last)
    : data_{first}, size_{static_cast<size_type>(last - first)}
    {}

    // contiguous container constructor
    // (uses anything with .data() and .size())
    template <typename Container>
    span(Container& c)
    : data_{c.data()}, size_{static_cast<size_type>(c.size())}
    {}

    // capacity functions

    constexpr __host__ __device__
    auto size(void) const noexcept -> size_type
    {
      return size_;
    }

    constexpr __host__ __device__
    auto length(void) const noexcept -> size_type
    {
      return size_;
    }

    // element access functions

    constexpr __host__ __device__
    auto data(void) const noexcept -> pointer
    {
      return data_;
    }

    constexpr __host__ __device__
    auto operator[](size_type const pos) const -> reference
    {
      return data_[pos];
    }
  };
}

#endif // REGULUS_VIEWS_SPAN_HPP_