#include <cstddef>
#include <thrust/device_vector.h>

#include "regulus/array.hpp"
#include "regulus/point.hpp"
#include "regulus/tetra.hpp"
#include "regulus/algorithm/orient.hpp"
#include "regulus/algorithm/fracture.hpp"

#include <catch.hpp>

TEST_CASE("Our fracture routine")
{
  SECTION("1-to-4 fracture")
  {
    auto const num_tetra = std::size_t{1};

    auto const pa = thrust::device_vector<std::ptrdiff_t>{{0, 1, 2, 3}};

    auto const tetra = regulus::tetra_t{0, 1, 2, 3};

  }
}