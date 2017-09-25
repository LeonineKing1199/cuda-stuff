#include <thrust/host_vector.h>

#include "regulus/views/span.hpp"
#include "regulus/array.hpp"

#include <catch.hpp>

TEST_CASE("Our span type")
{
  using int_span       = regulus::span<int>;
  using int_const_span = regulus::span<int const>;

  SECTION("should have be default constructible")
  {
    {
      int_span       s;
      int_const_span cs;

      REQUIRE((s.data() == nullptr && s.size() == 0));
      REQUIRE((cs.data() == nullptr && cs.size() == 0));
    }

    {
      int_span       s{};
      int_const_span cs{};

      REQUIRE((s.data()  == nullptr && s.size() == 0));
      REQUIRE((cs.data() == nullptr && cs.size() == 0));
    }

    {
      auto s  = int_span{};
      auto cs = int_const_span{};

      REQUIRE((s.data()  == nullptr && s.size() == 0));
      REQUIRE((cs.data() == nullptr && cs.size() == 0));
    }
  }

  SECTION("should be constructible from a pointer and a length")
  {
    {
      int  arr[4] = {1, 2, 3, 4};
      auto s      = int_const_span{&arr[0], 2};

      REQUIRE((s.length() == 2));
      REQUIRE((s.data()   == &arr[0]));

      REQUIRE((s[0] == 1 && s[1] == 2));
    }

    {
      int  arr[4] = {1, 2, 3, 4};
      auto s      = int_span{&arr[0], 2};

      REQUIRE((s.length() == 2));
      REQUIRE((s.data()   == &arr[0]));

      REQUIRE((s[0] == 1 && s[1] == 2));
    }

    {
      int* p = nullptr;
      auto s = int_span{p, static_cast<typename int_span::size_type>(0)};

      REQUIRE((s.data() == nullptr && s.length() == 0));
    }
  }

  SECTION("should be container-constructible")
  {
    auto i = regulus::array<int, 4>{1, 2, 3, 4};
    auto s = int_span{i};

    REQUIRE((s.size() == i.size() && s.data() == i.data()));
    REQUIRE((s[0] == 1 && s[1] == 2 && s[2] == 3 && s[3] == 4));
  }

  SECTION("should be range-constructible")
  {
    int a[4] = {1, 2, 3, 4};

    {
      auto* const begin = std::addressof(a[0]);
      auto* const end   = std::addressof(a[2]);

      auto s = int_span{begin, end};

      REQUIRE((s.data() == &a[0] && s.length() == 2));
      REQUIRE((s[0] == 1 && s[1] == 2));
    }

    {
      auto* const begin = std::addressof(a[0]);
      auto* const end   = std::addressof(a[0]);

      auto s = int_span{begin, end};

      REQUIRE((s.data() == begin && s.length() == 0));
    }

    {
      int* p = nullptr;

      auto s = int_span{p, p};
      REQUIRE((s.data() == nullptr && s.length() == 0));
    }
  }
}