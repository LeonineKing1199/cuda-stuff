#include "catch.hpp"
#include "array.hpp"
#include "maybe-int.hpp"

TEST_CASE("The maybe_int type")
{
  SECTION("should be constructible")
  {
    maybe_int<ptrdiff_t> const m{-1};
    REQUIRE(m == maybe_int<ptrdiff_t>{-1});

    maybe_int<int> const n{1337};
    maybe_int<int> const o{n};

    REQUIRE(n == o);
	}

  SECTION("should support equality operations")
  {
    maybe_int<int> const m = 1337;
    maybe_int<int> const n = 7331;

    // boost less_than_comparable2
    REQUIRE(m <= 7331);
    REQUIRE(n >= 1337);
    REQUIRE(7331 > m);
    REQUIRE(1337 < n);
    REQUIRE(1337 <= m);
    REQUIRE(7331 >= n);

    // boost equality_comparable2
    REQUIRE(1337 == m);
    REQUIRE(7331 != m);
    REQUIRE(n != 1337);

    // boost less_than_comparable1
    REQUIRE(n > m);
    REQUIRE(m <= n);
    REQUIRE(n >= m);

    // boost equality_comparable2
    REQUIRE(n != m);

    // top-level boost strong typedef comparisons
    REQUIRE(m == maybe_int<int>{1337});
    REQUIRE(m < n);
  }

  SECTION("should support array accessing properties")
  {
    array<int, 4> x = { 0, 1, 2, 3 };
    x[maybe_int<ptrdiff_t>{0}] = 1337;

    REQUIRE(x[0] == 1337);
  }

  SECTION("should support arithmetic operations")
  {
    maybe_int<int> const x = 14;

    REQUIRE(x * 2 == 28);
    REQUIRE(x / 2 == 7);
    REQUIRE(x + 3 == 17);
    REQUIRE(x - 3 == 11);
  }

  SECTION("should be convertible to a boolean")
  {
    maybe_int<ptrdiff_t> m = ptrdiff_t{-1};
    maybe_int<ptrdiff_t> n = ptrdiff_t{1337};

    REQUIRE(static_cast<bool>(m) == false);
    REQUIRE(static_cast<bool>(n) == true);
  }
}