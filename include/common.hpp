#ifndef REGULUS_COMMON_HPP_
#define REGULUS_COMMON_HPP_

#include <iostream>
#include <type_traits>

namespace reg
{
  template <bool B, typename T = void>
  using enable_if_t = typename std::enable_if<B, T>::type;
  
  template <
    typename T,
    typename = reg::enable_if_t<std::is_floating_point<T>::value>>
  struct point_type 
  {
    using type = T;
  };
  
  template <>
  struct point_type<float>
  {
    using type = float3;
  };
  
  template <>
  struct point_type<double>
  {
    using type = double3;
  };
  
  template <typename T>
  using point_t = typename point_type<T>::type;
}


auto operator==(reg::point_t<float> const& a, reg::point_t<float> const& b) -> bool;
auto operator==(reg::point_t<double> const& a, reg::point_t<double> const& b) -> bool;
auto operator<<(std::ostream& os, reg::point_t<float> const& p) ->  std::ostream&;
auto operator<<(std::ostream& os, reg::point_t<double> const& p) -> std::ostream&;

#endif // REGULUS_COMMON_HPP_

