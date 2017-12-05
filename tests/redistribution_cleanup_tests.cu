#include <cstddef>

#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>

#include "regulus/loc.hpp"
#include "regulus/array.hpp"

#include "regulus/algorithm/redistribution_cleanup.hpp"

#include <catch.hpp>

using std::size_t;
using std::ptrdiff_t;

namespace
{
  template <typename T, size_t N>
  auto operator==(
    thrust::host_vector<T> const& a, 
    regulus::array<T, N>   const& b) -> bool
  {
    for (auto i = 0; i < b.size(); ++i) {
      if (a[i] != b[i]) { return false; }
    }
    return true;
  }
}

TEST_CASE("Redistribution cleanup")
{
  SECTION("should filter out the appropriate content")
  {
    auto const pa_data = regulus::array<ptrdiff_t, 15>{0, 1, 2, 3, -1, 0, 2, 2, -1, 2, 3, -1, -1, 3, 3};
    auto const ta_data = regulus::array<ptrdiff_t, 15>{0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, -1, 0, 0};

    auto const la_data = regulus::array<regulus::loc_t, 15>{0};

    auto const nm_data = regulus::array<bool, 4>{true, false, false, false};

    auto pa = thrust::device_vector<ptrdiff_t>{pa_data.begin(), pa_data.end()};
    auto ta = thrust::device_vector<ptrdiff_t>{ta_data.begin(), ta_data.end()};
    auto la = thrust::device_vector<regulus::loc_t>{la_data.begin(), la_data.end()};
    auto nm = thrust::device_vector<bool>{nm_data.begin(), nm_data.end()};

    auto const assoc_size = regulus::redistribution_cleanup(pa, ta, la, nm);

    REQUIRE(assoc_size == 9);

    auto hpa = thrust::host_vector<ptrdiff_t>{pa};
    auto hta = thrust::host_vector<ptrdiff_t>{ta};
    auto hla = thrust::host_vector<regulus::loc_t>{la};

    thrust::sort(
      hpa.begin(), hpa.begin() + assoc_size, 
      thrust::less<ptrdiff_t>{});

    auto const expected_pa = regulus::array<ptrdiff_t, 15>{1, 2, 2, 2, 2, 3, 3, 3, 3, -1, -1, -1, -1, -1, -1};
    auto const expected_ta = regulus::array<ptrdiff_t, 15>{0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1};

    auto const expected_la = regulus::array<regulus::loc_t, 15>{0};

    REQUIRE(hpa == expected_pa);
    REQUIRE(hta == expected_ta);
    // REQUIRE(hla == expected_la);
  }
}