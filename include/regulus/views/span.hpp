#ifndef REGULUS_VIEWS_SPAN_HPP_
#define REGULUS_VIEWS_SPAN_HPP_

#include <cstddef>
#include <type_traits>

namespace regulus
{
  template <typename T>
  struct span
  {
  public:
    using value_type      = typename std::remove_cv<T>::type;
    using pointer         = value_type*;
    using const_pointer   = value_type const*;
    using reference       = value_type&;
    using const_reference = value_type const&;
    using size_type       = std::size_t;
    using iterator        = pointer;
    using const_iterator  = const_pointer;

  private:
    pointer   p_;
    size_type size_;

  public:

    // default constructor
    constexpr __host__ __device__
    span(void) noexcept
    : p_{nullptr}, size_{0}
    {}

    // pointer + size_type constructor
    constexpr __host__ __device__
    span(pointer const p, size_type const s)
    : p_{p}, size_{s}
    {}

    // pointer range constructor
    constexpr __host__ __device__
    span(pointer const first, pointer const last)
    : p_{first}, size_{static_cast<size_type>(last - first)}
    {}

    // contiguous container constructor
    // (uses anything with .data() and .size())
    template <
      typename Container,
      typename = typename std::enable_if<!std::is_same<Container, span>::value>::type>
    span(Container& c)
    : p_{c.data()}, size_{static_cast<size_type>(c.size())}
    {}

    // capacity functions

    // span::size
    constexpr __host__ __device__
    auto size(void) const noexcept -> size_type
    {
      return size_;
    }

    // span::length
    constexpr __host__ __device__
    auto length(void) const noexcept -> size_type
    {
      return size_;
    }

    // span::empty
    constexpr __host__ __device__
    auto empty(void) const noexcept -> bool
    {
      return size_ == 0;
    }

    // element access functions

    // span::data
    constexpr __host__ __device__
    auto data(void) const noexcept -> pointer
    {
      return p_;
    }

    // span::operator[]
    constexpr __host__ __device__
    auto operator[](size_type const pos) const -> reference
    {
      return p_[pos];
    }

    // span::front
    auto front(void) const noexcept -> reference
    {
      return p_[0];
    }

    // span::back
    auto back(void) const noexcept -> reference
    {
      return p_[size_ - 1];
    }

    // iterators

    // span::begin
    constexpr __host__ __device__
    auto begin(void) const noexcept -> iterator
    {
      return p_;
    }

    // span::cbegin
    constexpr __host__ __device__
    auto cbegin(void) const noexcept -> const_iterator
    {
      return p_;
    }

    // span::end
    constexpr __host__ __device__
    auto end(void) const noexcept -> iterator
    {
      return p_ + size_;
    }

    constexpr __host__ __device__
    auto cend(void) const noexcept -> const_iterator
    {
      return p_ + size_;
    }
  };

  template <typename T>
  auto make_span(T* p, std::size_t size) noexcept -> span<T>
  {
    return {p, size};
  }

  template <typename T>
  auto make_span(T* begin, T* end) noexcept -> span<T>
  {
    return {begin, static_cast<std::size_t>(end - begin)};
  }

  template <typename Container>
  auto make_span(Container& c) noexcept
  -> span<typename Container::value_type>
  {
    return {c.data(), c.size()};
  }
}

#endif // REGULUS_VIEWS_SPAN_HPP_