#include <utility>
#include <iostream>
#include <iterator>
#include <unordered_map>

#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "regulus/loc.hpp"
#include "regulus/algorithm/nominate.hpp"
#include "regulus/utils/make_rand_range.hpp"

#include <catch.hpp>

TEST_CASE("Our point nomination routine...")
{
  SECTION("should be able to handle larger point sets")
  {
    using regulus::loc_t;
    using int_t = std::ptrdiff_t;

    // we're trying to force a stupid high number of collisions
    // between tetrahedra and points to stress test how well the
    // algorithm is able to resolve potential conflicts
    auto const assoc_size = std::size_t{10'000};

    auto const min = int_t{0};
    auto const max = int_t{250};

    auto data_buffer = thrust::host_vector<int_t>{};
    data_buffer.reserve(2 * assoc_size);

    regulus::make_rand_range(
      (2 * assoc_size),
      min, max,
      std::back_inserter(data_buffer));

    REQUIRE(data_buffer.size() == (2 * assoc_size));

    auto ta = thrust::device_vector<int_t>{data_buffer.begin(), data_buffer.begin() + assoc_size};
    auto pa = thrust::device_vector<int_t>{data_buffer.begin() + assoc_size, data_buffer.end()};
    auto la = thrust::device_vector<loc_t>{assoc_size, regulus::outside_v};
    auto nm = thrust::device_vector<bool>{
      1 + static_cast<std::size_t>(*thrust::max_element(pa.begin(), pa.end())),
      false};

    regulus::nominate(pa, ta, la, nm);
    cudaDeviceSynchronize();

    auto h_ta = thrust::host_vector<std::ptrdiff_t>{ta};
    auto h_pa = thrust::host_vector<std::ptrdiff_t>{pa};
    auto h_nm = thrust::host_vector<bool>{nm};

    auto found_duplicate = false;
    auto num_nominated   = int{0};

    auto tet_pt_map = std::unordered_map<std::ptrdiff_t, std::ptrdiff_t>{};
    tet_pt_map.reserve(assoc_size);

    for (std::size_t i = 0; i < assoc_size; ++i) {
      auto const pa_id = h_pa[i];
      auto const ta_id = h_ta[i];

      if (h_nm[pa_id]) {
        auto const tet_pt = tet_pt_map.find(ta_id);

        if (tet_pt == tet_pt_map.end()) {
          tet_pt_map[ta_id] = pa_id;
          ++num_nominated;
        } else {
          if (pa_id != std::get<1>(*tet_pt)) {
            found_duplicate = true;
          }
        }
      }
    }

    REQUIRE(found_duplicate == false);
    REQUIRE(num_nominated > 0);
    REQUIRE(tet_pt_map.size() > 0);
  }
}