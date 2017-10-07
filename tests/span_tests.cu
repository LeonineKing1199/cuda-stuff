#include <array>
#include <algorithm>

#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>

#include "regulus/array.hpp"
#include "regulus/views/span.hpp" // <-- thing we're actually testing
#include "regulus/utils/make_rand_range.hpp"

#include <catch.hpp>

namespace
{
  // sort a device vector, dv, of integers
  // (non-blocking)
  auto device_sort(regulus::span<int> dv) -> void
  {
    thrust::sort(
      thrust::device,
      dv.begin(), dv.end());
  }

  // there is a thrust::is_sorted out there but to this extent,
  // we're also able to play with our span type and see how useful
  // taking a subspan can be
  auto is_sorted(
    regulus::span<int const> const dv) -> bool
  {
    if (dv.size() < 2) {
      return false;
    }

    auto const range_a = dv.subspan(0, dv.size() - 1);
    auto const range_b = dv.subspan(1, dv.size());

    auto const begin = thrust::make_zip_iterator(
      thrust::make_tuple(
        range_a.begin(), range_b.begin()));

    auto const end = begin + range_a.size();

    return range_a.size() == thrust::transform_reduce(
      thrust::device,
      begin, end,
      [] __device__ (thrust::tuple<int, int> const t) -> int
      {
        return (
          thrust::get<0>(t) <= thrust::get<1>(t) ? 1 : 0);
      },
      int{0},
      thrust::plus<int>{});
  }
}

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

  SECTION("should support Thrust to some extent")
  {
    auto const init = [](void) -> thrust::host_vector<int>
    {
      auto const num_vals = std::size_t{1000};
      auto const min      = int{0};
      auto const max      = int{1000};

      auto tmp_buffer = thrust::host_vector<int>{num_vals, -1};
      regulus::make_rand_range(
        num_vals,
        min, max,
        tmp_buffer.begin());
      return tmp_buffer;
    };

    auto data = thrust::device_vector<int>{init()};
    auto v    = regulus::make_span(data);

    REQUIRE(v.size() == 1000);
    REQUIRE(!is_sorted(v));

    device_sort(v);

    REQUIRE(is_sorted(v));
  }
}