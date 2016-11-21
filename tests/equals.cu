#include "gtest/gtest.h"
#include "math/equals.hpp"

TEST(EqualsFunction, ShouldCompareAccurately)
{
  EXPECT_EQ(true, eq(0.6f, 3.0f / 5));
  EXPECT_EQ(true, eq(0.1, 1.0 / 10));
  EXPECT_EQ(false, eq(0.1, 1.1 / 10));
}

TEST(RoundToFunction, ShouldRoundNumbersAppropriately)
{
  EXPECT_EQ(true, eq(0.333, round_to(1.0 / 3, 3)));
  EXPECT_EQ(true, eq(3.14285714, round_to(22.0 / 7, 8)));
}
