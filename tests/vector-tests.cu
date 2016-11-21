#include "gtest/gtest.h"
#include "math/vector.hpp"

TEST(VectorType, ShouldBeProductive)
{
  vector<double, 3> const a = { 1, 2, 3 };
  vector<double, 3> const b = { 2, 2, 2 };
  
  // we get 2 + 4 + 6 => 12
  EXPECT_EQ(12, a * b);
}
