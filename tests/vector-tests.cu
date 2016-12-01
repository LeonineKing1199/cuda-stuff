#include "catch.hpp"
#include "math/vector.hpp"

TEST_CASE("Our vector type implementation")
{
  vector<double, 3> const a = { 1, 2, 3 };
  vector<double, 3> const b = { 2, 2, 2 };
  
  // we get 2 + 4 + 6 => 12
  REQUIRE(12 == (a * b));
}
