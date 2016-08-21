#ifndef REGULUS_MATH_HPP_
#define REGULUS_MATH_HPP_

#include <array>
#include <type_traits>
#include <cmath>
#include <algorithm>
#include <functional>
#include <numeric>

#include "point.hpp"

template <int N>
struct greater_than_three
  : std::integral_constant<bool, (N > 3)>
{};

template <int N>
struct greater_than_zero
  : std::integral_constant<bool, (N > 0)>
{};

#endif // REGULUS_MATH_HPP_
