#include <vector>
#include <iterator>
#include <iostream>
#include "regulus/utils/make_rand_range.hpp"

#include <catch.hpp>

TEST_CASE("Random number generation")
{
  SECTION("should at least compile well")
  {
    auto const num_vals = size_t{16};

    auto v = std::vector<ptrdiff_t>{};
    v.reserve(num_vals);

    regulus::make_rand_range<ptrdiff_t>(
      num_vals,
      0, 128,
      std::back_inserter(v));

    REQUIRE(v.size() == num_vals);
  }
}