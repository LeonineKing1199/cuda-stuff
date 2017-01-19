#include "math/tetra.hpp"
#include <iostream>
#include <type_traits>

auto operator<<(std::ostream& os, orientation o) -> std::ostream&
{
  os << static_cast<std::underlying_type<orientation>::type>(o);
  return os;
}

auto operator==(int4 const& a, int4 const& b) -> bool
{
  return (a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w);
}

auto operator!=(int4 const& a, int4 const& b) -> bool
{
  return (a.x != b.x) || (a.y != b.y) || (a.z != b.z) || (a.w != b.w); 
}

auto operator<<(std::ostream& os, int4 const& t) -> std::ostream&
{
  os << "{ " << t.x << " " << t.y << " " << t.z << " " << t.w << " }";
  return os;
}