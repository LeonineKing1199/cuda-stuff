#include "gtest/gtest.h"
#include "math/point.hpp"
#include "math/tetra.hpp"

struct tetra_type : public testing::Test
{
public:
  using real = double;
  using point_f = point_t<real>;
  
  point_f const a;
  point_f const b;
  point_f const c;
  point_f const d;
  
  tetra_type(void) 
  : a{ 0, 0, 0 },
    b{ 9, 0, 0 },
    c{ 0, 9, 0 },
    d{ 0, 0, 9 }
  {}
};

TEST_F(tetra_type, global_initialization)
{
  EXPECT_TRUE((a == point_f{ 0, 0, 0 }));
  EXPECT_TRUE((b == point_f{ 9, 0, 0 }));
  EXPECT_TRUE((c == point_f{ 0, 9, 0 }));
  EXPECT_TRUE((d == point_f{ 0, 0, 9 }));
}

TEST_F(tetra_type, should_yield_orientation)
{
  EXPECT_EQ(orientation::positive, orient<real>(a, b, c, d));
}

TEST_F(tetra_type, should_work_with_insphere_stuff_as_well)
{
  point_f const x{ 3, 3, 3 };
  point_f const y{ 1000, 1000, 1000 };
  point_f const z = b;
  
  EXPECT_TRUE((orientation::negative == insphere<real>(a, b, c, d, x)));
  EXPECT_TRUE((orientation::positive == insphere<real>(a, b, c, d, y)));
  EXPECT_TRUE((orientation::zero == insphere<real>(a, b, c, d, z)));
}

TEST_F(tetra_type, all_the_location_code_testing)
{
    // We should be able to accurately determine all 6 edge intersections
    {
      point_f const e10{ 4.5, 0.0, 0.0 };
      point_f const e20{ 0.0, 4.5, 0.0 };
      point_f const e30{ 0.0, 0.0, 4.5 };
      point_f const e21{ 4.5, 4.5, 0.0 };
      point_f const e31{ 4.5, 0.0, 4.5 };
      point_f const e23{ 0.0, 4.5, 4.5 };
            
      EXPECT_TRUE((
        eq<real>(det(matrix<real, 4, 4>{ 1, 0, 0, 0,
                                         1, 0, 9, 0,
                                         1, 0, 0, 9,
                                         1, 4.5, 0, 0 }), 364.5)));
                                         
      EXPECT_TRUE(orient<real>(d, c, b, e10) == orientation::positive);
      EXPECT_TRUE(orient<real>(a, c, d, e10) == orientation::positive);
      EXPECT_TRUE(orient<real>(a, d, b, e10) == orientation::zero);
      EXPECT_TRUE(orient<real>(a, b, c, e10) == orientation::zero);
      
      EXPECT_TRUE(loc<real>(a, b, c, d, e10) == 3);
      EXPECT_TRUE(loc<real>(a, b, c, d, e20) == 5);
      EXPECT_TRUE(loc<real>(a, b, c, d, e30) == 9);
      EXPECT_TRUE(loc<real>(a, b, c, d, e21) == 6);
      EXPECT_TRUE(loc<real>(a, b, c, d, e31) == 10);
      EXPECT_TRUE(loc<real>(a, b, c, d, e23) == 12);
    }
    
    // We should be able to determine all 4 face intersections
    {
      point_f const f321{ 3, 3, 3 };
      point_f const f023{ 0, 4.5, 3 };
      point_f const f031{ 4.5, 0, 3 };
      point_f const f012{ 3, 3, 0 };
      
      EXPECT_TRUE(loc<real>(a, b, c, d, f321) == 14);
      EXPECT_TRUE(loc<real>(a, b, c, d, f023) == 13);
      EXPECT_TRUE(loc<real>(a, b, c, d, f031) == 11);
      EXPECT_TRUE(loc<real>(a, b, c, d, f012) == 7);
    }
    
    // We should be able to determine all 4 vertex intersections
    {
      point_f const v0 = a;
      point_f const v1 = b;
      point_f const v2 = c;
      point_f const v3 = d;
      
      EXPECT_TRUE(loc<real>(a, b, c, d, v0) == 1);
      EXPECT_TRUE(loc<real>(a, b, c, d, v1) == 2);
      EXPECT_TRUE(loc<real>(a, b, c, d, v2) == 4);
      EXPECT_TRUE(loc<real>(a, b, c, d, v3) == 8);
    }
    
    // We should be able to determine if a point is inside a tetrahedron
    {
      point_f const p{ 1, 1, 1 };
      
      EXPECT_TRUE(loc<real>(a, b, c, d, p) == 15);
    }
    
    // We should be able to determine if a point is outside a tetrahedron
    {
      point_f const p{ 3.01, 3.01, 3.01 };
      
      EXPECT_TRUE(loc<real>(a, b, c, d, p) == -1);
    }
}
