#include <iostream>

#include "../include/common.hpp"

auto operator==(reg::point_t<float> const& a, reg::point_t<float> const& b) -> bool
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

auto operator==(reg::point_t<double> const& a, reg::point_t<double> const& b) -> bool
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

auto operator<<(std::ostream& os, reg::point_t<float> const& p) -> std::ostream&
{
  os << p.x << " " << p.y << " " << p.z;
  return os;
}

auto operator<<(std::ostream& os, reg::point_t<double> const& p) -> std::ostream&
{
  os << p.x << " " << p.y << " " << p.z;
  return os;
}
