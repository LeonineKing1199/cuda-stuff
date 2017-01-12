#include "catch.hpp"
#include "index_t.hpp"
#include "lib/fract-locations.hpp"

#include <array>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using thrust::host_vector;
using thrust::device_vector;

template <
  typename Container1,
  typename Container2
>
auto compare_ranges(Container1 const& c1, Container2 const& c2) -> bool
{
  bool equal = true;

  auto it2 = c2.begin();
  for (auto it1 = c1.begin(); it1 != c1.end() && equal; ++it1, ++it2) {
    equal = equal && (*it1 == *it2);
  }
  return equal;
}

TEST_CASE("The fract locations routine")
{
  typename index_t::value_type const tmp_pa_data[9] = {0, 0, 0, 1, 2, 2, 3, 3, 3};
  typename loc_t::value_type const tmp_la_data[9] = {3, 3, 3, 15, 7, 7, 3, 3, 3};
  unsigned const tmp_nm_data[4] = {1, 1, 0, 1};

  device_vector<index_t> pa{tmp_pa_data, tmp_pa_data + 9};
  device_vector<loc_t> la{tmp_la_data, tmp_la_data + 9};
  device_vector<unsigned> nm{tmp_nm_data, tmp_nm_data + 4};

  device_vector<index_t> fl{pa.size(), -1};

  fract_locations(pa.size(), pa, nm, la, fl);

  std::array<index_t, 9> const expected_vals = {1, 2, 3, 6, 6, 6, 7, 8, 9};
  host_vector<index_t> fl_copy = fl;

  REQUIRE(compare_ranges(expected_vals, fl_copy));
}