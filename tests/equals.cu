#include "catch.hpp"
#include "math/equals.hpp"

TEST_CASE("The equals function")
{
  REQUIRE(eq(0.6f, 3.0f / 5));
  REQUIRE(eq(0.1, 1.0 / 10));
  REQUIRE(!eq(0.1, 1.1 / 10));
}

TEST_CASE("The round_to function")
{
  REQUIRE(eq(0.333, round_to(1.0 / 3, 3)));
  REQUIRE(eq(3.14285714, round_to(22.0 / 7, 8)));
}
