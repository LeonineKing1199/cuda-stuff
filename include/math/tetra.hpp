#ifndef REGULUS_TETRA_HPP_
#define REGULUS_TETRA_HPP_

#include "../common.hpp"
#include "matrix.hpp"
#include "equals.hpp"

template <typename T>
__host__ __device__
auto orient(
  reg::point_t<T> const& a,
  reg::point_t<T> const& b,
  reg::point_t<T> const& c,
  reg::point_t<T> const& d)
-> int
{                                      
  auto const det = m.det(matrix<T, 4, 4>{ 1.0, a.x, a.y, a.z,
                                          1.0, b.x, b.y, b.z,
                                          1.0, c.x, c.y, c.z,
                                          1.0, d.x, d.y, d.z });
  
  auto const not_equal_to_zero = !eq(det, 0.0);
}

#endif // REGULUS_TETRA_HPP_
