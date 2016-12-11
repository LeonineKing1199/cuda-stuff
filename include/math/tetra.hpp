#ifndef REGULUS_TETRA_HPP_
#define REGULUS_TETRA_HPP_

#include "point.hpp"
#include "../array.hpp"
#include "matrix.hpp"
#include "equals.hpp"

using tetra = int4;

enum class orientation { positive = 1, zero = 0, negative = 2 };

template <typename T>
__host__ __device__
auto make_matrix(
  point_t<T> const& a,
  point_t<T> const& b,
  point_t<T> const& c,
  point_t<T> const& d) -> matrix<T, 4, 4>
{
  return { 1, a.x, a.y, a.z,
           1, b.x, b.y, b.z,
           1, c.x, c.y, c.z,
           1, d.x, d.y, d.z };
}

// Routine that calculates whether the point d
// is above the triangle spanned by abc
template <typename T>
__host__ __device__
auto orient(
  point_t<T> const& a,
  point_t<T> const& b,
  point_t<T> const& c,
  point_t<T> const& d)
-> orientation
{ 
  matrix<T, 4, 4> const m = make_matrix<T>(a, b, c, d);

  T const det_value = det<T, 4>(m);
  auto const not_equal_to_zero = !eq<T>(det_value, 0.0);
  
  if (det_value > 0.0 && not_equal_to_zero) {
    return orientation::positive;
    
  } else if (!not_equal_to_zero) {
    return orientation::zero;
    
  } else {
    return orientation::negative;
  }
}

// Calculate the magnitude of a point (i.e. vector)
template <typename T>
__host__ __device__
auto mag(point_t<T> const& p) -> T
{
  return (
    p.x * p.x +
    p.y * p.y +
    p.z * p.z);
}

// Function that calculates whether or not p is contained
// in the sphere circumscribed by the tetrahedron abcd
template <typename T>
__host__ __device__
auto insphere(
  point_t<T> const& a,
  point_t<T> const& b,
  point_t<T> const& c,
  point_t<T> const& d,
  point_t<T> const& p)
-> orientation
{
  matrix<T, 5, 5> const m{
    1.0, a.x, a.y, a.z, mag<T>(a),
    1.0, b.x, b.y, b.z, mag<T>(b),
    1.0, c.x, c.y, c.z, mag<T>(c),
    1.0, d.x, d.y, d.z, mag<T>(d),
    1.0, p.x, p.y, p.z, mag<T>(p) };
    
  auto const det_value = det<T, 5>(m);
  auto const not_equal_to_zero = !eq<T>(det_value, 0.0);
  
  if (det_value > 0.0 && not_equal_to_zero) {
    return orientation::positive;
    
  } else if (!not_equal_to_zero) {
    return orientation::zero;
    
  } else {
    return orientation::negative;
  } 
}

// Function that builds a location code which is a bitwise
// encoding of a point's location relative to the tetrahedron
// spanned by abcd
//
// bit value of 0 = orientation::zero
// bit value of 1 = orientation::positive
// loc() == 255? then point is outside abcd
//
// All faces are positively oriented
// bit loc 0 = face 0 => 321 
// bit loc 1 = face 1 => 023
// bit loc 2 = face 2 => 031
// bit loc 3 = face 3 => 012
template <typename T>
__host__ __device__
auto loc(
  point_t<T> const& a,
  point_t<T> const& b,
  point_t<T> const& c,
  point_t<T> const& d,
  point_t<T> const& p
) -> unsigned char
{
  return 0;
  /*matrix<int, 4, 4> const face_ids = { 3, 2, 1,
                                       0, 2, 3,
                                       0, 3, 1,
                                       0, 1, 2 };
  
  int const num_pts{4};
  array<point_t<T>, num_pts> const pts{a, b, c, d};
  
  int loc{0};
  
  for (int i = 0; i < num_pts; ++i) {
    orientation const ort = orient<T>(
      pts[face_ids[i][0]],
      pts[face_ids[i][1]],
      pts[face_ids[i][2]],
      p);
    
    if (ort == orientation::negative) {
      return -1;
    }
    
    loc |= (ort << i);
  }
  
  return loc;//*/
}


// Determine the volume of the tetrahedron spanned
// by points a, b, c, d
template <typename T>
__host__ __device__
auto vol(
  point_t<T> const& a,
  point_t<T> const& b,
  point_t<T> const& c,
  point_t<T> const& d) -> T
{
  auto const m = make_matrix<T>(a, b, c, d);
  
  return absolute(det<T>(m)) / 6;
}

#endif // REGULUS_TETRA_HPP_
