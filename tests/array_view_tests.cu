#include <iterator>

#include <thrust/sort.h>
#include <thrust/equal.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "regulus/array.hpp"
#include "regulus/views/array_view.hpp"
#include "regulus/utils/make_rand_range.hpp"

#include <catch.hpp>

TEST_CASE("Our array view type")
{
  SECTION("should be constructible")
  {
    auto x = regulus::array<int, 4>{2, 1, 0, 3};
    auto v = regulus::array_view<int>{x};

    REQUIRE(v.size() == 4);

    thrust::sort(thrust::host, v.begin(), v.end());
    auto const y = regulus::array<int, 4>{0, 1, 2, 3};
    REQUIRE(x == y);
  }

  SECTION("we should be able to sort a device view")
  {
    auto const assoc_size = std::size_t{1000};
    auto const min        = std::ptrdiff_t{0};
    auto const max        = std::ptrdiff_t{128};

    auto h_data = thrust::host_vector<std::ptrdiff_t>{};
    h_data.reserve(assoc_size);

    REQUIRE(h_data.size() == 0);
    REQUIRE(h_data.capacity() == assoc_size);

    regulus::make_rand_range(
      assoc_size,
      min, max,
      std::back_inserter(h_data));

    REQUIRE(h_data.size() == assoc_size);
  }
}