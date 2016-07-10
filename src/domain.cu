#include <iostream>

#include "../include/domain.hpp"

bool operator==(reg::point_t<float> const& a, reg::point_t<float> const& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

bool operator==(reg::point_t<double> const& a, reg::point_t<double> const& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

std::ostream& operator<<(std::ostream& os, reg::point_t<float> const& p)
{
  os << p.x << " " << p.y << " " << p.z;
  return os;
}

std::ostream& operator<<(std::ostream& os, reg::point_t<double> const& p)
{
  os << p.x << " " << p.y << " " << p.z;
  return os;
}

