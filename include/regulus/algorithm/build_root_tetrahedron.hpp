#ifndef REGULUS_ALGORITHM_BUILD_ROOT_TETRAHEDRON_HPP_
#define REGULUS_ALGORITHM_BUILD_ROOT_TETRAHEDRON_HPP_

#include <type_traits>

#include <thrust/reduce.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>

#include "regulus/array.hpp"
#include "regulus/point_traits.hpp"
#include "regulus/is_point.hpp"
#include "regulus/utils/make_point.hpp"

namespace regulus 
{
  namespace detail 
  {
    template <typename Point>
    struct add_points
      : public thrust::binary_function<
          Point const, Point const, 
          Point
        >
    {
      __host__ __device__
      auto operator()(Point const a, Point const b) -> Point
      { 
        return make_point<Point>(
          a.x + b.x, 
          a.y + b.y, 
          a.z + b.z); 
      }
    };

    template <typename Point>
    struct calc_radius_from 
      : public thrust::unary_function<
          Point const, 
          typename point_traits<Point>::value_type
        >
    {
      Point const p;

      calc_radius_from(void) = delete;
      calc_radius_from(Point const pp)
        : p(pp)
      {}

      __host__ __device__
      auto operator()(Point const q)     
      -> typename point_traits<Point>::value_type
      {
        return sqrt(
          pow(q.x - p.x, 2) + 
          pow(q.y - p.y, 2) + 
          pow(q.z - p.z, 2));
      }
    };
  } // detail

  template <
    typename Point, 
    typename InputIterator,
    typename = typename std::enable_if<
      is_point<Point>::value &&
      std::is_same<
        Point, 
        thrust::iterator_traits<InputIterator>::value_type
      >::value
    >::type
  >
  auto build_root_tetrahedron(
    InputIterator begin, InputIterator end)
  -> array<Point, 4>
  {
    using coord_value_type = typename point_traits<Point>::value_type;

    // first thing we do is calculate the centroid of the point set
    // sum all points first
    auto const point_sum = thrust::reduce(
      begin, end,
      make_point<Point>(0, 0, 0),
      detail::add_points<Point>{});

    auto const num_points = thrust::distance(begin, end);

    // then divide by the total number of points
    auto const centroid = make_point<Point>(
      point_sum.x / num_points,
      point_sum.y / num_points,
      point_sum.z / num_points);

    // we're now ready to find out which point in our set is the farthest from
    // the centroid
    auto const calc_radius_begin = thrust::make_transform_iterator(
      begin, detail::calc_radius_from<Point>{centroid});

    auto const calc_radius_end = calc_radius_begin + num_points;

    auto const radius = thrust::reduce(
      calc_radius_begin, calc_radius_end,
      coord_value_type{0},
      thrust::maximum<coord_value_type>{});

    // and now we're ready to build our 4 vertices!
    auto const x = sqrt(3) * radius;

    return {
      make_point<Point>(-x,  x, -x),
      make_point<Point>(-x, -x,  x),
      make_point<Point>( x, -x, -x),
      make_point<Point>( x,  x,  x)
    };
  }

} // regulus

#endif // REGULUS_ALGORITHM_BUILD_ROOT_TETRAHEDRON_HPP_