#include <utility>

#include <thrust/fill.h>
#include <thrust/tuple.h>
#include <thrust/remove.h>
#include <thrust/distance.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <thrust/iterator/zip_iterator.h>

#include "regulus/algorithm/redistribution_cleanup.hpp"

namespace regulus
{
  /**
   * Unfortunately, point redistribution requires some manual
   * cleanup of the arrays.
   *
   * Namely, because writing the association data doesn't leave
   * perfectly contiguous valid chunks of data, we need to first
   * coalesce everything into being a single span and then we need
   * to remove all the nominated tuples, i.e. any tuple with a
   * nm[pa[id]] == true
   *
   * This function then returns the new size of the assocation arrays
   */

  auto redistribution_cleanup(
    span<std::ptrdiff_t> const pa,
    span<std::ptrdiff_t> const ta,
    span<regulus::loc_t> const la,
    span<bool const>     const nm) -> std::ptrdiff_t
  {
    auto zip_begin = thrust::make_zip_iterator(
      thrust::make_tuple(
        pa.begin(),
        ta.begin(),
        la.begin()));

    auto zip_end = zip_begin + pa.size();

    auto new_zip_end = thrust::remove_if(
      thrust::device,
      zip_begin, zip_end,
      [=] __device__ (auto&& pa_ta_la) -> bool
      {
        using thrust::get;
        auto const pa_id = get<0>(std::forward<decltype(pa_ta_la)>(pa_ta_la));
        if (pa_id < 0) {
          return true;
        } else {
          return nm[pa_id];
          // return false;
        }
      });

    thrust::fill(
      thrust::device,
      new_zip_end, zip_end,
      thrust::make_tuple(-1, -1, regulus::outside_v));

    cudaDeviceSynchronize();

    auto const assoc_size = thrust::distance(zip_begin, new_zip_end);
    return assoc_size;
  }
}