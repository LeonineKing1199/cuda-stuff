#ifndef REGULUS_TETRA_HPP_
#define REGULUS_TETRA_HPP_

#include "../common.hpp"
#include "matrix.hpp"
#include "equals.hpp"

enum class orientation { positive, zero, negative };

template <typename T>
__host__ __device__
auto orient(
  reg::point_t<T> const& a,
  reg::point_t<T> const& b,
  reg::point_t<T> const& c,
  reg::point_t<T> const& d)
-> orientation
{ 
  matrix<T, 4, 4> const m{ 1, a.x, a.y, a.z,
                           1, b.x, b.y, b.z,
                           1, c.x, c.y, c.z,
                           1, d.x, d.y, d.z };

  auto const det = m.det();
  auto const not_equal_to_zero = !eq<T>(det, 0.0);
  
  if (det > 0.0 && not_equal_to_zero) {
    return orientation::positive;
    
  } else if (!not_equal_to_zero) {
    return orientation::zero;
    
  } else {
    return orientation::negative;
  }
}

template <typename T>
__host__ __device__
auto mag(reg::point_t<T> const& p) -> T
{
  return (
    p.x * p.x +
    p.y * p.y +
    p.z * p.z);
}

template <typename T>
__host__ __device__
auto insphere(
  reg::point_t<T> const& a,
  reg::point_t<T> const& b,
  reg::point_t<T> const& c,
  reg::point_t<T> const& d,
  reg::point_t<T> const& p)
-> orientation
{
  matrix<T, 5, 5> const m{
    1.0, a.x, a.y, a.z, mag<T>(a),
    1.0, b.x, b.y, b.z, mag<T>(b),
    1.0, c.x, c.y, c.z, mag<T>(c),
    1.0, d.x, d.y, d.z, mag<T>(d),
    1.0, p.x, p.y, p.z, mag<T>(p) };
    
  auto const det = m.det();
  auto const not_equal_to_zero = !eq<T>(det, 0.0);
  
  if (det > 0.0 && not_equal_to_zero) {
    return orientation::positive;
    
  } else if (!not_equal_to_zero) {
    return orientation::zero;
    
  } else {
    return orientation::negative;
  }
}

#endif // REGULUS_TETRA_HPP_
