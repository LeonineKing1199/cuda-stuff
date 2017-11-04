#include <cstddef>
#include <thrust/device_vector.h>

#include "regulus/array.hpp"
#include "regulus/algorithm/mark_nominated_tetra.hpp"

#include <catch.hpp>

TEST_CASE("Marking which tetrahedra were nominated...")
{
  SECTION("...should work as intended")
  {
    using std::ptrdiff_t;

    auto const ta_data = regulus::array<ptrdiff_t, 9>{0, 1, 2, 0, 3, 4, 5, 6, 5};
    auto const pa_data = regulus::array<ptrdiff_t, 9>{0, 0, 0, 1, 1, 2, 2, 2, 3};
    auto const nm_data = regulus::array<bool, 4>{true, false, true, false};

    auto const ta = thrust::device_vector<ptrdiff_t>{ta_data.begin(), ta_data.end()};
    auto const pa = thrust::device_vector<ptrdiff_t>{pa_data.begin(), pa_data.end()};
    auto const nm = thrust::device_vector<bool>{nm_data.begin(), nm_data.end()};

    auto nt = thrust::device_vector<ptrdiff_t>{7, -1};

    regulus::mark_nominated_tetra(ta, pa, nm, nt);
    cudaDeviceSynchronize();

    auto h_nt = thrust::host_vector<ptrdiff_t>{nt};

    REQUIRE((h_nt[0] ==  0));
    REQUIRE((h_nt[1] ==  1));
    REQUIRE((h_nt[2] ==  2));
    REQUIRE((h_nt[3] == -1));
    REQUIRE((h_nt[4] ==  5));
    REQUIRE((h_nt[5] ==  6));
    REQUIRE((h_nt[6] ==  7));
  }
}