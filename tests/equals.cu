#include "regulus/utils/equals.hpp"
#include "regulus/utils/round_to.hpp"

#include <catch.hpp>

TEST_CASE("The eq function")
{
  REQUIRE(regulus::eq(0.6f, 3.0f / 5));
  REQUIRE(regulus::eq(0.1, 1.0 / 10));
  REQUIRE(!regulus::eq(0.1, 1.1 / 10));
}

TEST_CASE("The round_to function")
{
  REQUIRE(regulus::eq(0.333, regulus::round_to(1.0 / 3, 3)));
  REQUIRE(regulus::eq(3.14285714, regulus::round_to(22.0 / 7, 8)));
}
