#include "regulus/tetra_t.hpp"
#include "regulus/access.hpp"
#include "regulus/point_t.hpp"
#include "regulus/algorithm/orient.hpp"
#include "regulus/algorithm/location.hpp"
#include "regulus/algorithm/insphere.hpp"

#include <catch.hpp>

TEST_CASE("Our tetrahedral implementation")
{
  SECTION("Out default tetrahedron type should be settable and gettable")
  {
    using regulus::set;
    using regulus::get;

    auto tetra = regulus::tetra_t{};

    set<0>(tetra, 0);
    set<1>(tetra, 1);
    set<2>(tetra, 2);
    set<3>(tetra, 3);

    REQUIRE(get<0>(tetra) == 0);
    REQUIRE(get<1>(tetra) == 1);
    REQUIRE(get<2>(tetra) == 2);
    REQUIRE(get<3>(tetra) == 3);
  }

  using regulus::orientation;

  using real    = double;
  using point_t = regulus::point_t<real>;
  
  auto const a = point_t{0, 0, 0};
  auto const b = point_t{9, 0, 0};
  auto const c = point_t{0, 9, 0};
  auto const d = point_t{0, 0, 9};

  SECTION("should give us its orientation")
  {
    REQUIRE(orient(a, b, c, d) == orientation::positive);
  }

  SECTION("should support insphere tests")
  {
    auto const x = point_t{3, 3, 3};
    auto const y = point_t{1000, 1000, 1000};
    auto const z = b;
    
    REQUIRE(orientation::negative == insphere(a, b, c, d, x));
    REQUIRE(orientation::positive == insphere(a, b, c, d, y));
    REQUIRE(orientation::zero     == insphere(a, b, c, d, z));    
  }

  SECTION("should support location code testing")
  {
    
    // We should be able to accurately determine all 6 edge intersections
    {      
      auto const e10 = point_t{4.5, 0.0, 0.0};
      auto const e20 = point_t{0.0, 4.5, 0.0};
      auto const e30 = point_t{0.0, 0.0, 4.5};
      auto const e21 = point_t{4.5, 4.5, 0.0};
      auto const e31 = point_t{4.5, 0.0, 4.5};
      auto const e23 = point_t{0.0, 4.5, 4.5};
                                               
      REQUIRE(loc(a, b, c, d, e10) == 3);
      REQUIRE(loc(a, b, c, d, e20) == 5);
      REQUIRE(loc(a, b, c, d, e30) == 9);
      REQUIRE(loc(a, b, c, d, e21) == 6);
      REQUIRE(loc(a, b, c, d, e31) == 10);
      REQUIRE(loc(a, b, c, d, e23) == 12);
    }
    
    // We should be able to determine all 4 face intersections
    {
      auto const f321 = point_t{3, 3, 3};
      auto const f023 = point_t{0, 4.5, 3};
      auto const f031 = point_t{4.5, 0, 3};
      auto const f012 = point_t{3, 3, 0};
      
      REQUIRE(loc(a, b, c, d, f321) == 14);
      REQUIRE(loc(a, b, c, d, f023) == 13);
      REQUIRE(loc(a, b, c, d, f031) == 11);
      REQUIRE(loc(a, b, c, d, f012) == 7);
    }
    
    // We should be able to determine all 4 vertex intersections
    {
      auto const v0 = a;
      auto const v1 = b;
      auto const v2 = c;
      auto const v3 = d;
      
      REQUIRE(loc(a, b, c, d, v0) == 1);
      REQUIRE(loc(a, b, c, d, v1) == 2);
      REQUIRE(loc(a, b, c, d, v2) == 4);
      REQUIRE(loc(a, b, c, d, v3) == 8);
    }
    
    // We should be able to determine if a point is inside a tetrahedron
    {
      auto const p = point_t{1, 1, 1};
      
      REQUIRE(loc(a, b, c, d, p) == 15);
    }
    
    // We should be able to determine if a point is outside a tetrahedron
    {
      auto const p = point_t{3.01, 3.01, 3.01};
      REQUIRE(loc(a, b, c, d, p) == UINT8_MAX);
    }
  }
}


