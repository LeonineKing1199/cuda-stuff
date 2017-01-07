#include "catch.hpp"
#include "math/point.hpp"
#include "math/tetra.hpp"

TEST_CASE("Our tetrahedral implementation")
{
  using real = double;
  using point_f = point_t<real>;
  
  point_f const a{ 0, 0, 0 };
  point_f const b{ 9, 0, 0 };
  point_f const c{ 0, 9, 0 };
  point_f const d{ 0, 0, 9 };

  REQUIRE((a == point_f{ 0, 0, 0 }));
  REQUIRE((b == point_f{ 9, 0, 0 }));
  REQUIRE((c == point_f{ 0, 9, 0 }));
  REQUIRE((d == point_f{ 0, 0, 9 }));  

  SECTION("should give us its orientation")
  {
    REQUIRE((orientation::positive == orient<real>(a, b, c, d)));
  }

  SECTION("should support insphere tests")
  {
    point_f const x{ 3, 3, 3 };
    point_f const y{ 1000, 1000, 1000 };
    point_f const z = b;
    
    REQUIRE((orientation::negative == insphere<real>(a, b, c, d, x)));
    REQUIRE((orientation::positive == insphere<real>(a, b, c, d, y)));
    REQUIRE((orientation::zero == insphere<real>(a, b, c, d, z)));    
  }

  SECTION("should support location code testing")
  {
    // We should be able to accurately determine all 6 edge intersections
    {
      point_f const e10{ 4.5, 0.0, 0.0 };
      point_f const e20{ 0.0, 4.5, 0.0 };
      point_f const e30{ 0.0, 0.0, 4.5 };
      point_f const e21{ 4.5, 4.5, 0.0 };
      point_f const e31{ 4.5, 0.0, 4.5 };
      point_f const e23{ 0.0, 4.5, 4.5 };
            
      REQUIRE((
        eq<real>(det(matrix<real, 4, 4>{ 1, 0, 0, 0,
                                         1, 0, 9, 0,
                                         1, 0, 0, 9,
                                         1, 4.5, 0, 0 }), 364.5)));
                                         
      REQUIRE(orient<real>(d, c, b, e10) == orientation::positive);
      REQUIRE(orient<real>(a, c, d, e10) == orientation::positive);
      REQUIRE(orient<real>(a, d, b, e10) == orientation::zero);
      REQUIRE(orient<real>(a, b, c, e10) == orientation::zero);
      
      REQUIRE(loc<real>(a, b, c, d, e10) == loc_t{3});
      REQUIRE(loc<real>(a, b, c, d, e20) == loc_t{5});
      REQUIRE(loc<real>(a, b, c, d, e30) == loc_t{9});
      REQUIRE(loc<real>(a, b, c, d, e21) == loc_t{6});
      REQUIRE(loc<real>(a, b, c, d, e31) == loc_t{10});
      REQUIRE(loc<real>(a, b, c, d, e23) == loc_t{12});
    }
    
    // We should be able to determine all 4 face intersections
    {
      point_f const f321{ 3, 3, 3 };
      point_f const f023{ 0, 4.5, 3 };
      point_f const f031{ 4.5, 0, 3 };
      point_f const f012{ 3, 3, 0 };
      
      REQUIRE(loc<real>(a, b, c, d, f321) == loc_t{14});
      REQUIRE(loc<real>(a, b, c, d, f023) == loc_t{13});
      REQUIRE(loc<real>(a, b, c, d, f031) == loc_t{11});
      REQUIRE(loc<real>(a, b, c, d, f012) == loc_t{7});
    }
    
    // We should be able to determine all 4 vertex intersections
    {
      point_f const v0 = a;
      point_f const v1 = b;
      point_f const v2 = c;
      point_f const v3 = d;
      
      REQUIRE(loc<real>(a, b, c, d, v0) == loc_t{1});
      REQUIRE(loc<real>(a, b, c, d, v1) == loc_t{2});
      REQUIRE(loc<real>(a, b, c, d, v2) == loc_t{4});
      REQUIRE(loc<real>(a, b, c, d, v3) == loc_t{8});
    }
    
    // We should be able to determine if a point is inside a tetrahedron
    {
      point_f const p{ 1, 1, 1 };
      
      REQUIRE(loc<real>(a, b, c, d, p) == loc_t{15});
    }
    
    // We should be able to determine if a point is outside a tetrahedron
    {
      point_f const p{ 3.01, 3.01, 3.01 };
      
      REQUIRE(loc<real>(a, b, c, d, p) == loc_t{-1});
    }
  }
}


