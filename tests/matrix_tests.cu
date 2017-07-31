#include "regulus/matrix.hpp"
#include "regulus/point.hpp"
#include "regulus/utils/equals.hpp"
#include "regulus/utils/round_to.hpp"

#include <catch.hpp>

TEST_CASE("Our matrix implementation")
{
  SECTION("should be comparable and accessible")
  {
    using coord_type = float;

    auto const a = coord_type{1.0};
    auto const b = coord_type{2.0};
    auto const c = coord_type{3.0};
    auto const d = coord_type{4.0};
    auto const e = coord_type{5.0};
    auto const f = coord_type{6.0};

    // aggregate constructor
    auto const m = regulus::matrix<coord_type, 2, 3>
      {a, b, c, d, e, f};

    // test for size() function
    REQUIRE(6 == m.size());

    // make sure that operator== works
    REQUIRE((m == regulus::matrix<coord_type, 2, 3>{a, b, c, d, e, f}));

    auto const not_m = regulus::matrix<coord_type, 2, 3>
      {a, b, c, d, e, 7.0};

    REQUIRE(m != not_m);

    // then our row() and col() functions should operate
    // as we expect
    REQUIRE((m.row(0) == regulus::vector<coord_type, 3>{a, b, c}));
    REQUIRE((m.row(1) == regulus::vector<coord_type, 3>{d, e, f}));

    REQUIRE((m.col(0) == regulus::vector<coord_type, 2>{a, d}));
    REQUIRE((m.col(1) == regulus::vector<coord_type, 2>{b, e}));
    REQUIRE((m.col(2) == regulus::vector<coord_type, 2>{c, f}));
  }

  SECTION("should support multiplication")
  {
    regulus::matrix<float, 2, 3> const a{1.0f, 2.0f, 3.0f,
                                         4.0f, 5.0f, 6.0f};

    regulus::matrix<float, 3, 2> const b{ 7.0f,  8.0f,
                                          9.0f, 10.0f,
                                         11.0f, 12.0f};

    regulus::matrix<float, 2, 2> const c{ 58.0f,  64.0f,
                                         139.0f, 154.0f};

    REQUIRE(c == (a * b));
  }

  SECTION("should support determinant operations")
  {
    using regulus::eq;
    using regulus::round_to;

    auto const t = regulus::matrix<float, 4, 4>{
      1.0f, 0.0f, 0.0f, 0.0f,
      1.0f, 9.0f, 0.0f, 0.0f,
      1.0f, 0.0f, 9.0f, 0.0f,
      1.0f, 0.0f, 0.0f, 9.0f};

    REQUIRE(eq(det(t), 729.0f));

    auto const r = regulus::matrix<double, 4, 4>{
       0.0, 1.85, 0.63, 2.65,
      1.92, 1.57, 1.15, 2.94,
       2.7, 2.45, 0.57, 2.81,
      2.33, 1.68,  1.0, 0.05};

    REQUIRE(eq(round_to(det(r), 3), -10.928));

    auto const u = regulus::matrix<float, 4, 4>{
      1.0, 0.0, 0.0, 0.0,
      1.0, 9.0, 0.0, 0.0,
      1.0, 0.0, 9.0, 0.0,
      1.0, 3.0, 3.0, 0.0};

    REQUIRE(eq(det(u), 0.0f));
  }
}