#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include "regulus/array.hpp"
#include "regulus/algorithm/location.hpp"
#include "regulus/algorithm/nominate.hpp"

#include <catch.hpp>

TEST_CASE("Nominating points...")
{
  SECTION("should work as expected")
  {
    auto const assoc_size = size_t{11};

    using array_t = regulus::array<ptrdiff_t, assoc_size>;

    auto const ta_data = array_t{0, 1, 2, 3, 2, 5, 6, 7, 8, 1, 8};
    auto const pa_data = array_t{0, 0, 0, 0, 2, 2, 3, 3, 3, 4, 4};

    auto ta = thrust::device_vector<ptrdiff_t>{assoc_size, -1};
    auto pa = thrust::device_vector<ptrdiff_t>{assoc_size, -1};
    auto la = thrust::device_vector<regulus::loc_t>{assoc_size, regulus::outside_v};

    auto const copy = [](
      auto const begin,
      auto const end,
      auto output)
    {
      thrust::copy(thrust::seq, begin, end, output);
    };

    copy(ta_data.begin(), ta_data.end(), ta.begin());
    copy(pa_data.begin(), pa_data.end(), pa.begin());

    // because pa_data holds array indices, we can somewhat
    // assume for an initial point set that the number
    // of points will be the largest index (i.e. back of the array)
    // and so the number of elements is 1 more than that
    // this is largely sufficient for our nomination scheme
    // and the size of nm will be determined by the largets value
    // in pa
    auto const num_pts = 1 + thrust::max_element(
      pa_data.begin(),
      pa_data.end());


  }
}