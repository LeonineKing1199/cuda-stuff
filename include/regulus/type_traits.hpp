#ifndef REGULUS_TYPE_TRAITS_HPP_
#define REGULUS_TYPE_TRAITS_HPP_

#include <type_traits>

namespace regulus
{
  template <bool B, typename T = void>
  using enable_if_t = typename std::enable_if<B, T>::type;

  template <typename T, typename U>
  inline constexpr
  bool is_same_v = std::is_same<T, U>::value;

  // sample impl of void_t from cppreference.com
  namespace detail
  {
    template <typename ...Ts>
    struct make_void
    {
      using type = void;
    };
  }

  template <typename ...Ts>
  using void_t = typename detail::make_void<Ts...>::type;
}

#endif // REGULUS_TYPE_TRAITS_HPP_