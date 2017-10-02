#include <array>
#include <algorithm>
#include <thrust/host_vector.h>

#include "regulus/views/span.hpp"
#include "regulus/array.hpp"

#include <catch.hpp>

TEST_CASE("Our span type")
{
  using int_span       = regulus::span<int>;
  using int_const_span = regulus::span<int const>;

  SECTION("should be default constructible")
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

  SECTION("should support copy, move and assign semantics")
  {
    auto s1 = int_span{};
    REQUIRE(s1.empty());

    auto arr = std::array<int, 3>{3, 4, 5};

    auto s2 = int_const_span{arr};
    REQUIRE((s2.length() == 3 && s2.data() == arr.data()));

    s2 = s1;
    REQUIRE(s2.empty());

    s2 = int_const_span{arr};
    REQUIRE((s2.length() == 3 && s2.data() == arr.data()));

    auto x = int_span{arr};
    auto y = std::move(x);
    REQUIRE((y.data() == arr.data() && y.size() == arr.size()));
  }

  SECTION("make_span")
  {
    auto arr = regulus::array<int, 4>{1, 2, 3, 4};

    auto s1 = regulus::make_span(arr.data(), arr.size());
    auto s2 = regulus::make_span(arr.begin(), arr.end());
    auto s3 = regulus::make_span(arr);

    REQUIRE((s1.data() == arr.data() && s1.size() == arr.size()));
    REQUIRE((s2.data() == arr.data() && s2.size() == arr.size()));
    REQUIRE((s3.data() == arr.data() && s3.size() == arr.size()));

    auto s4 = regulus::make_span(s1);

    REQUIRE((s4.data() == arr.data() && s4.size() == arr.size()));

    auto const val = int{1337};

    s4[0] = val;

    REQUIRE((s1[0] == val && s2[0] == val && s3[0] == val));
  }

  SECTION("make_const_span")
  {
    {
      // using a const container
      auto const arr = regulus::array<int, 4>{1, 2, 3, 4};

      auto s1 = regulus::make_const_span(arr.data(), arr.size());
      auto s2 = regulus::make_const_span(arr.begin(), arr.end());
      auto s3 = regulus::make_const_span(arr);

      REQUIRE((s1.data() == arr.data() && s1.size() == arr.size()));
      REQUIRE((s2.data() == arr.data() && s2.size() == arr.size()));
      REQUIRE((s3.data() == arr.data() && s3.size() == arr.size()));

      auto s4 = regulus::make_const_span(s1);

      REQUIRE((s4.data() == arr.data() && s4.size() == arr.size()));

      auto const val = int{1};

      REQUIRE((s1[0] == val && s2[0] == val && s3[0] == val && s4[0] == val));
    }

    {
      // using a mutable container
      auto arr = regulus::array<int, 4>{1, 2, 3, 4};

      auto s1 = regulus::make_const_span(arr.data(), arr.size());
      auto s2 = regulus::make_const_span(arr.begin(), arr.end());
      auto s3 = regulus::make_const_span(arr);

      REQUIRE((s1.data() == arr.data() && s1.size() == arr.size()));
      REQUIRE((s2.data() == arr.data() && s2.size() == arr.size()));
      REQUIRE((s3.data() == arr.data() && s3.size() == arr.size()));

      auto s4 = regulus::make_const_span(s1);

      REQUIRE((s4.data() == arr.data() && s4.size() == arr.size()));

      auto const val = int{1};

      REQUIRE((s1[0] == val && s2[0] == val && s3[0] == val && s4[0] == val));
    }
  }

  SECTION("front and back access")
  {
    auto data = regulus::array<int, 4>{1, 2, 3, 4};
    auto s    = regulus::make_span(data);

    REQUIRE(s.front() == 1);
    REQUIRE(s.back()  == 4);

    s.front() = 1337;
    s.back()  = 7331;

    REQUIRE((s[0] == 1337 && s[3] == 7331));
  }

  SECTION("begin/end traversal")
  {
    auto arr     = regulus::array<int, 4>{3, 2, 4, 1};
    auto const s = regulus::make_span(arr);

    std::sort(s.begin(), s.end());

    REQUIRE((arr == regulus::array<int, 4>{1, 2, 3, 4}));

    auto const cs = int_const_span{arr};

    REQUIRE((*cs.cbegin() == 1));
    REQUIRE((cs.cbegin() + cs.size() == cs.cend()));
  }


}