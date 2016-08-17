#ifndef REGULUS_COMMON_HPP_
#define REGULUS_COMMON_HPP_

#include <iostream>
#include <type_traits>
#include <cfloat>

// helper template that emulates C++14 functionality
template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

// our main point class
// as a sane default the encapsulated typedef is whatever
// was passed in as a template argument
template <
  typename T,
  typename = enable_if_t<std::is_floating_point<T>::value>>
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

auto operator==(point_t<float> const& a, point_t<float> const& b) -> bool;
auto operator==(point_t<double> const& a, point_t<double> const& b) -> bool;
auto operator<<(std::ostream& os, point_t<float> const& p) ->  std::ostream&;
auto operator<<(std::ostream& os, point_t<double> const& p) -> std::ostream&;

#endif // REGULUS_COMMON_HPP_

