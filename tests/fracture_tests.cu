#include <cstddef>
#include <thrust/device_vector.h>

#include "regulus/loc.hpp"
#include "regulus/point.hpp"
#include "regulus/tetra.hpp"
#include "regulus/globals.hpp"
#include "regulus/algorithm/fracture.hpp"
#include "regulus/algorithm/fract_locations.hpp"

#include <catch.hpp>

TEST_CASE("Our fracture routine...")
{
  using std::size_t;
  using std::ptrdiff_t;

  SECTION("... should work for a basic 1-to-4 fracture")
  {
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
    mesh[0]   = regulus::tetra_t{0, 1, 2, 3};

    regulus::fracture(
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

  SECTION(".. should work for a 2-to-6 flip")
  {
    auto const num_tetra = 2;

    auto const insert_idx = ptrdiff_t{1337};

    auto pa = thrust::device_vector<ptrdiff_t>{2};

    pa[0] = insert_idx;
    pa[1] = insert_idx;

    auto ta = thrust::device_vector<ptrdiff_t>{2};

    ta[0] = 0;
    ta[1] = 1;

    auto la = thrust::device_vector<regulus::loc_t>{2};

    la[0] = 7;
    la[1] = 7;

    auto mesh = thrust::device_vector<regulus::tetra_t>{6};

    mesh[0] = regulus::tetra_t{0, 1, 2, 3};
    mesh[1] = regulus::tetra_t{2, 1, 0, 4};

    auto fl = thrust::device_vector<ptrdiff_t>{2, -1};

    auto nm = thrust::device_vector<bool>{insert_idx + 1, false};
    nm[insert_idx] = true;

    regulus::fract_locations(pa, la, nm, fl);
    regulus::fracture(num_tetra, pa, ta, la, nm, fl, mesh);
    cudaDeviceSynchronize();

    auto const h_mesh = thrust::host_vector<regulus::tetra_t>{mesh};

    REQUIRE((h_mesh[0] == regulus::tetra_t{3, 2, 1, insert_idx}));
    REQUIRE((h_mesh[1] == regulus::tetra_t{4, 0, 1, insert_idx}));
    REQUIRE((h_mesh[2] == regulus::tetra_t{0, 2, 3, insert_idx}));
    REQUIRE((h_mesh[3] == regulus::tetra_t{0, 3, 1, insert_idx}));
    REQUIRE((h_mesh[4] == regulus::tetra_t{2, 0, 4, insert_idx}));
    REQUIRE((h_mesh[5] == regulus::tetra_t{2, 4, 1, insert_idx}));
  }
}