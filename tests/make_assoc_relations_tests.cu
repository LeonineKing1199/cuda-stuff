#include <iostream>

#ifdef _MSC_VER
#pragma warning(push)

// thrust/iterator/iterator_adaptor.h(203)
// : warning C4244: '+=': conversion from '__int64' to 'int', possible loss of data
#pragma warning(disable: 4244)

#include <iterator>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#pragma warning(pop)

#elif

#include <iterator>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#endif


#include "regulus/point_t.hpp"
#include "regulus/algorithm/build_root_tetrahedron.hpp"
#include "regulus/algorithm/make_assoc_relations.hpp"
#include "regulus/utils/gen_cartesian_domain.hpp"

#include <catch.hpp>

TEST_CASE("Making the initial set of association relations should work")
{
    SECTION("Cartesian set")
    {
        using point_t = double3;

        auto const grid_length = size_t{9};

        auto h_pts = thrust::host_vector<point_t>{};
        h_pts.reserve(grid_length * grid_length * grid_length);

        regulus::gen_cartesian_domain<point_t>(
            grid_length,
            std::back_inserter(h_pts));

        REQUIRE(h_pts.size() == (grid_length * grid_length * grid_length));

        auto d_pts = thrust::device_vector<point_t>{h_pts};

        auto const root_vertices = regulus::build_root_tetrahedron<point_t>(
            d_pts.begin(),
            d_pts.end());

        auto const num_assocs = 4 * d_pts.size();

        auto pa = thrust::device_vector<ptrdiff_t>{num_assocs, -1};
        auto ta = thrust::device_vector<ptrdiff_t>{num_assocs, -1};
        auto la = thrust::device_vector<uint8_t>{num_assocs, UINT8_MAX};

        regulus::make_assoc_relations<point_t>(
          root_vertices,
          d_pts,
          pa, ta, la);

        cudaDeviceSynchronize();

        auto h_pa = thrust::host_vector<ptrdiff_t>{pa};
        auto h_ta = thrust::host_vector<ptrdiff_t>{ta};
        auto h_la = thrust::host_vector<uint8_t>{la};

        for (decltype(h_pa.size()) i = 0; i < h_pa.size(); ++i) {
          if (h_pa[i] == -1) { break; }

          std::cout << "{ " << h_pa[i] << ", " << h_ta[i] << ", " << static_cast<unsigned>(h_la[i]) << " }\n";
        }
    }
}