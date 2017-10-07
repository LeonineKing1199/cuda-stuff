#ifndef REGULUS_VIEWS_SPAN_HPP_
#define REGULUS_VIEWS_SPAN_HPP_

#include <cstddef>

namespace regulus
{
  template <typename T>
  struct span
  {
  public:
    using value_type      = T;
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
    span(pointer const p, size_type const s) noexcept
    : p_{p}, size_{s}
    {}

    // pointer range constructor
    constexpr __host__ __device__
    span(pointer const begin, pointer const end)
    : p_{begin}, size_{static_cast<size_type>(end - begin)}
    {}

    // thrust::device_vector constructor
    template <typename T>
    span(thrust::device_vector<T>& dv)
    : p_{dv.data().get()}, size_{dv.size()}
    {}

    // contiguous container constructor
    // (uses anything with .data() and .size())
    template <
      typename Container,
      typename = typename std::enable_if<!std::is_same<Container, span>::value>::type>
    span(Container& c)
    : p_{c.data()}, size_{static_cast<size_type>(c.size())}
    {}

    template <
      typename Container,
      typename = typename std::enable_if<!std::is_same<Container, span>::value>::type>
    span(Container const& c)
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

    // span::cend
    constexpr __host__ __device__
    auto cend(void) const noexcept -> const_iterator
    {
      return p_ + size_;
    }

    // span mutations

    constexpr __host__ __device__
    auto subspan(
      size_type const begin,
      size_type const end) const -> span
    {
      return {p_ + begin, p_ + end};
    }
  };

  // make_span()

  template <typename T>
  auto make_span(
    T*          const p,
    std::size_t const size) noexcept -> span<T>
  {
    return {p, size};
  }

  template <typename T>
  auto make_span(
    T* const begin,
    T* const end) noexcept -> span<T>
  {
    return {begin, static_cast<std::size_t>(end - begin)};
  }

  template <typename Container>
  auto make_span(Container& c) noexcept
  -> span<typename Container::value_type>
  {
    return {c};
  }

  // make_const_span

  template <typename T>
  auto make_const_span(
    T const*    const p,
    std::size_t const size) noexcept -> span<T const>
  {
    return {p, size};
  }

  template <typename T>
  auto make_const_span(
    T const* const begin,
    T const* const end) noexcept -> span<T const>
  {
    return {begin, static_cast<std::size_t>(end - begin)};
  }

  template <typename Container>
  auto make_const_span(Container const& c) noexcept
  -> span<typename Container::value_type const>
  {
    return {c};
  }
}

#endif // REGULUS_VIEWS_SPAN_HPP_