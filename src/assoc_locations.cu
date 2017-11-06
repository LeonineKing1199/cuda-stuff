#include <thrust/functional.h>
#include <thrust/transform_scan.h>
#include <thrust/execution_policy.h>

#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include "regulus/algorithm/assoc_locations.hpp"

using std::ptrdiff_t;

namespace regulus
{
  auto assoc_locations(
    span<ptrdiff_t const> const ta,
    span<ptrdiff_t const> const nt,
    span<ptrdiff_t>       const al) -> void
  {
    using thrust::get;

    auto const zip_begin = thrust::make_zip_iterator(
      thrust::make_tuple(
        ta.begin(),
        thrust::make_counting_iterator<ptrdiff_t>(0)));

    auto const zip_end = zip_begin + ta.size();

    thrust::transform_inclusive_scan(
      thrust::device,
      zip_begin, zip_end,
      al.begin(),
      [=] __device__ (auto const ta_tid_pair) -> ptrdiff_t
      {
        auto const ta_id    = get<0>(ta_tid_pair);
        auto const tuple_id = get<1>(ta_tid_pair);

        auto const nominated_tuple_id = nt[ta_id];

        auto const is_tetra_fractured  = (nominated_tuple_id >= 0);
        auto const is_fracturing_tuple = (tuple_id == nominated_tuple_id);

        return (
          is_tetra_fractured && !is_fracturing_tuple
          ? ptrdiff_t{4}
          : ptrdiff_t{0});
      },
      thrust::plus<ptrdiff_t>{});
  }
}