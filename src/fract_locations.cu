#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/transform_scan.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>

#include "regulus/loc.hpp"
#include "regulus/views/span.hpp"
#include "regulus/algorithm/fract_locations.hpp"

namespace regulus
{
  auto fract_locations(
    span<std::ptrdiff_t const> const pa,
    span<loc_t          const> const la,
    span<bool           const> const nm,
    span<std::ptrdiff_t>       const fl) -> void
  {
    using thrust::get;

    auto const zip_begin = thrust::make_zip_iterator(
      thrust::make_tuple(
        pa.begin(),
        la.begin()));

    thrust::transform_inclusive_scan(
      thrust::device,
      zip_begin, zip_begin + pa.size(),
      fl.begin(),
      [=] __device__ (auto const pa_la_pair) -> std::ptrdiff_t
      {
        return nm[get<0>(pa_la_pair)]
          ? __popc(get<1>(pa_la_pair))
          : 0;
      },
      thrust::plus<std::ptrdiff_t>{});
  }
}
