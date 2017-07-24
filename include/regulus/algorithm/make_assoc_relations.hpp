#ifndef REGULUS_ALGORITHM_MAKE_ASSOC_RELATIONS_HPP_
#define REGULUS_ALGORITHM_MAKE_ASSOC_RELATIONS_HPP_

#include <thrust/tuple.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include "regulus/array.hpp"
#include "regulus/is_point.hpp"
#include "regulus/type_traits.hpp"
#include "regulus/algorithm/location.hpp"

namespace regulus
{
  namespace detail
  {
    template <typename Point>
    struct get_loc_code
      : public thrust::unary_function<
          thrust::tuple<Point, size_t> const&,
          thrust::tuple<ptrdiff_t, ptrdiff_t, uint8_t>
    >
    {
      Point const a;
      Point const b;
      Point const c;
      Point const d;

      get_loc_code(void) = delete;
      get_loc_code(
        Point const aa,
        Point const bb,
        Point const cc,
        Point const dd)
      : a(aa), b(bb), c(cc), d(dd)
      {}

      __host__ __device__
      auto operator()(thrust::tuple<Point, size_t> const& pt_and_index)
      -> thrust::tuple<ptrdiff_t, ptrdiff_t, uint8_t>
      {
        using thrust::get;

        return thrust::make_tuple(
          get<1>(pt_and_index),
          0,
          loc(
            a, b, c, d,
            get<0>(pt_and_index)));
      }
    };
  }

  template <
    typename Point,
    typename = enable_if_t<is_point<Point>::value>
  >
  auto make_assoc_relations(
    regulus::array<Point, 4>         const& vtx,
    thrust::device_vector<Point>     const& pts,
    thrust::device_vector<ptrdiff_t>      & pa,
    thrust::device_vector<ptrdiff_t>      & ta,
    thrust::device_vector<uint8_t>        & la)
  -> void
  {
    auto const begin = thrust::make_zip_iterator(
      thrust::make_tuple(
        pts.begin(),
        thrust::make_counting_iterator(size_t{0})));

    auto zip_output = thrust::make_zip_iterator(
      thrust::make_tuple(
        pa.begin(),
        ta.begin(),
        la.begin()));

    thrust::transform(
      begin, begin + pts.size(),
      zip_output,
      detail::get_loc_code<Point>{
        vtx[0],
        vtx[1],
        vtx[2],
        vtx[3]});
  }
} // regulus


#endif // REGULUS_ALGORITHM_MAKE_ASSOC_RELATIONS_HPP_