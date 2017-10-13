#include <cstddef>

#include <thrust/equal.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "regulus/loc.hpp"
#include "regulus/array.hpp"
#include "regulus/views/span.hpp"
#include "regulus/algorithm/fract_locations.hpp"

#include <catch.hpp>

TEST_CASE("The fract locations routine")
{
  auto const tmp_pa_data = regulus::array<std::ptrdiff_t const, 9>{0, 0, 0,  1, 2, 2, 3, 3, 3};
  auto const tmp_la_data = regulus::array<regulus::loc_t const, 9>{3, 3, 3, 15, 7, 7, 3, 3, 3};

  auto const tmp_nm_data = regulus::array<bool, 4>{1, 1, 0, 1};

  auto const pa = thrust::device_vector<std::ptrdiff_t>{tmp_pa_data.begin(), tmp_pa_data.end()};
  auto const la = thrust::device_vector<regulus::loc_t>{tmp_la_data.begin(), tmp_la_data.end()};

  auto const nm = thrust::device_vector<bool>{tmp_nm_data.begin(), tmp_nm_data.end()};

  auto fl = thrust::device_vector<std::ptrdiff_t>{pa.size(), -1};

  using regulus::make_span;
  using regulus::make_const_span;

  regulus::fract_locations(
    make_const_span(pa),
    make_const_span(la),
    make_const_span(nm),
    make_span(fl));
  cudaDeviceSynchronize();

  auto const expected_vals = regulus::array<std::ptrdiff_t const, 9>{ 1, 2, 3, 6, 6, 6, 7, 8, 9 };
  auto const fl_copy       = thrust::host_vector<std::ptrdiff_t>{fl};

  REQUIRE((
    thrust::equal(
      expected_vals.begin(), expected_vals.end(),
      fl_copy.begin())));
}