#ifndef REGULUS_VIEWS_ARRAY_VIEW_HPP_
#define REGULUS_VIEWS_ARRAY_VIEW_HPP_

#include <cstddef>

namespace regulus
{
  template <typename T>
  struct array_view
  {
    using value_type      = T;
    using pointer         = T*;
    using reference       = T&;
    using iterator        = T*;
    using const_pointer   = T const*;
    using const_reference = T const&;
    using const_iterator  = T const*;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;

    pointer   const data_;
    size_type const size_;

    constexpr array_view(void)              noexcept : data_{nullptr}, size_{0} {};
    constexpr array_view(array_view const&) noexcept = default;

    template <typename Container>
    __host__ __device__
    array_view(Container& c)
    : data_{c.data()}, size_{c.size()}
    {}

    __host__ __device__
    auto begin(void) noexcept -> iterator
    { return data_; }

    __host__ __device__
    auto end(void) noexcept -> iterator
    { return data_ + size_; }

    __host__ __device__
    auto cbegin(void) noexcept -> const_iterator
    { return data_; }

    __host__ __device__
    auto cend(void) noexcept -> const_iterator
    { return data_ + size_; }

    __host__ __device__
    auto size(void) noexcept -> size_type
    { return size_; }

    __host__ __device__
    auto data(void) noexcept -> pointer
    { return data_; }
  };
}

#endif // REGULUS_VIEWS_ARRAY_VIEW_HPP_