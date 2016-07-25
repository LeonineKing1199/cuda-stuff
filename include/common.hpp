#ifndef REGULUS_COMMON_HPP_
#define REGULUS_COMMON_HPP_

#include <iostream>
#include <type_traits>
#include <cfloat>

namespace reg
{
  // helper template that emulates C++14 functionality
  template <bool B, typename T = void>
  using enable_if_t = typename std::enable_if<B, T>::type;
  
  // our main point class
  // as a sane default the encapsulated typedef is whatever
  // was passed in as a template argument
  template <
    typename T,
    typename = reg::enable_if_t<std::is_floating_point<T>::value>>
  struct point_type 
  {
    using type = T;
  };
  
  // our floating point specialization
  template <>
  struct point_type<float>
  {
    using type = float3;
    constexpr float static const max_coord_value = FLT_MAX;
  };
  
  // our double floating point specialization
  template <>
  struct point_type<double>
  {
    using type = double3;
    constexpr double static const max_coord_value = DBL_MAX;
  };
  
  // a convenience templated type alias helper
  template <typename T>
  using point_t = typename point_type<T>::type;
  
  // our tetrahedral structure
  struct tetra_t
  {
    int4 v;
  };
}


auto operator==(reg::point_t<float> const& a, reg::point_t<float> const& b) -> bool;
auto operator==(reg::point_t<double> const& a, reg::point_t<double> const& b) -> bool;
auto operator<<(std::ostream& os, reg::point_t<float> const& p) ->  std::ostream&;
auto operator<<(std::ostream& os, reg::point_t<double> const& p) -> std::ostream&;

#endif // REGULUS_COMMON_HPP_

