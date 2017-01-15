#include "math/tetra.hpp"
#include <iostream>
#include <type_traits>

auto operator<<(std::ostream& os, orientation o) -> std::ostream&
{
  os << static_cast<std::underlying_type<orientation>::type>(o);
  return os;
}