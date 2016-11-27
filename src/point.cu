#include "math/point.hpp"
#include <iostream>

__host__ __device__
auto operator==(point_t<float> const& a, point_t<float> const& b) -> bool
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

__host__ __device__
auto operator==(point_t<double> const& a, point_t<double> const& b) -> bool
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

auto operator<<(std::ostream& os, point_t<float> const& p) -> std::ostream&
{
  os << p.x << " " << p.y << " " << p.z;
  return os;
}

auto operator<<(std::ostream& os, point_t<double> const& p) -> std::ostream&
{
  os << p.x << " " << p.y << " " << p.z;
  return os;
}
