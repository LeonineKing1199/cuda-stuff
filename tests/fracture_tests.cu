#include <cstddef>
#include <thrust/device_vector.h>

#include "regulus/loc.hpp"
#include "regulus/point.hpp"
#include "regulus/tetra.hpp"
#include "regulus/globals.hpp"
#include "regulus/algorithm/fracture.hpp"

#include <catch.hpp>

TEST_CASE("Our fracture routine...")
{
  SECTION("... should work for a basic 1-to-4 fracture")
  {
    using std::size_t;
    using std::ptrdiff_t;

    auto const num_tetra = size_t{1};

    auto pa = thrust::device_vector<ptrdiff_t>{1};
    pa[0]   = 4; // index in the point buffer of insertion vertex

    auto ta = thrust::device_vector<ptrdiff_t>{1};
    ta[0]   = 0;

    auto la = thrust::device_vector<regulus::loc_t>{1};
    la[0]   = 15;

    auto nm = thrust::device_vector<bool>{5, false};
    nm[4]   = true;

    auto fl = thrust::device_vector<ptrdiff_t>{1};
    fl[0]   = 3;

    auto mesh = thrust::device_vector<regulus::tetra_t>{4};
    mesh[0] = regulus::tetra_t{0, 1, 2, 3};

    fracture_kernel<<<regulus::bpg, regulus::tpb>>>(
      num_tetra,
      pa, ta, la,
      nm, fl,
      mesh);
    cudaDeviceSynchronize();

    auto h_mesh = thrust::host_vector<regulus::tetra_t>{mesh};

    REQUIRE((h_mesh[0] == regulus::tetra_t{3, 2, 1, 4}));
    REQUIRE((h_mesh[1] == regulus::tetra_t{0, 2, 3, 4}));
    REQUIRE((h_mesh[2] == regulus::tetra_t{0, 3, 1, 4}));
    REQUIRE((h_mesh[3] == regulus::tetra_t{0, 1, 2, 4}));
  }

  SECTION("... should work with a significantly larger data set")
  {

  }
}