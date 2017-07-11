#include <catch.hpp>

#include "regulus/vector.hpp"

TEST_CASE("Our vector type implementation")
{
  regulus::vector<double, 3> const a = {1, 2, 3};
  regulus::vector<double, 3> const b = {2, 2, 2};
  
  // we get 2 + 4 + 6 => 12
  REQUIRE(12 == (a * b));
}
