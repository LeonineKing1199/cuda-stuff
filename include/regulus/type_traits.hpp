#ifndef REGULUS_TYPE_TRAITS_HPP_
#define REGULUS_TYPE_TRAITS_HPP_

#include <type_traits>

namespace regulus
{
  template <typename T, typename U>
  inline constexpr
  bool is_same_v = std::is_same<T, U>::value;

  template <typename T>
  inline constexpr
  bool is_arithmetic_v = std::is_arithmetic<T>::value;

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

  template <typename Point>
  struct is_point : std::integral_constant<
    bool,
    std::is_arithmetic<
      typename std::decay<decltype(std::declval<Point>().x)>::type
    >::value
    &&
    std::is_same<
      typename std::decay<decltype(std::declval<Point>().x)>::type,
      typename std::decay<decltype(std::declval<Point>().y)>::type
    >::value
    &&
    std::is_same<
      typename std::decay<decltype(std::declval<Point>().x)>::type,
      typename std::decay<decltype(std::declval<Point>().z)>::type
    >::value
  >
  {};

  template <typename Point>
  inline constexpr
  bool is_point_v = is_point<Point>::value;
}

#endif // REGULUS_TYPE_TRAITS_HPP_