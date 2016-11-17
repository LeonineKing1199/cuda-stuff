//#include "tests/test-suite.hpp"

#include "gtest/gtest.h"

auto factorial(int const n) -> int
{
  if (n == 0) {
    return 1;
  } else {
    return -1;
  }
}

TEST(FactorialTest, HandlesZeroInput) {
  EXPECT_EQ(1, factorial(0));
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
