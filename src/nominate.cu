#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/iterator_traits.h>

#include "regulus/algorithm/nominate.hpp"

using zip_tuple_t  = thrust::tuple<ptrdiff_t, ptrdiff_t, regulus::loc_t>;
using ta_pa_pair_t = thrust::tuple<ptrdiff_t, ptrdiff_t>;

namespace
{
  struct sort_by_ta : public thrust::binary_function<zip_tuple_t const,
                                                     zip_tuple_t const,
                                                     bool>
  {
    __host__ __device__
    auto operator()(zip_tuple_t const a, zip_tuple_t const b) -> bool
    {
      using thrust::get;

      auto const a_ta = get<0>(a);
      auto const b_ta = get<0>(b);

      auto const a_pa = get<1>(a);
      auto const b_pa = get<1>(b);

      return (
        a_ta == b_ta
        ? (a_pa < b_pa)
        : (a_ta < b_ta));
    }
  };

  struct unique_by_ta : public thrust::binary_function<ta_pa_pair_t const,
                                                       ta_pa_pair_t const,
                                                       bool>
  {
    __host__ __device__
    auto operator()(ta_pa_pair_t const a, ta_pa_pair_t const b) -> bool
    {
      using thrust::get;

      auto const a_ta = get<0>(a);
      auto const b_ta = get<0>(b);

      return a_ta == b_ta;
    }
  };
}

namespace regulus
{
  auto nominate(
    // effective size of the associations arrays (pa, ta, la)
    size_t const assoc_size,
    thrust::device_vector<ptrdiff_t>& pa,
    thrust::device_vector<ptrdiff_t>& ta,
    thrust::device_vector<loc_t>    & la,
    thrust::device_vector<bool>     & nm) -> void
  {
    auto pa_copy = thrust::device_vector<ptrdiff_t>{assoc_size, -1};
    auto ta_copy = thrust::device_vector<ptrdiff_t>{assoc_size, -1};

    auto zip_begin = thrust::make_zip_iterator(
      thrust::make_tuple(
        ta.begin(),
        pa.begin(),
        la.begin()));

    auto pair_begin = thrust::make_zip_iterator(
      thrust::make_tuple(
        ta.begin(),
        pa.begin()));

    auto pair_copy_begin = thrust::make_zip_iterator(
      thrust::make_tuple(
        ta_copy.begin(),
        pa_copy.begin()));

    thrust::sort(
      zip_begin, zip_begin + assoc_size,
      sort_by_ta{});

    auto pair_copy_end = thrust::unique_copy(
      pair_begin, pair_begin + assoc_size,
      pair_copy_begin,
      unique_by_ta{});
  }
}