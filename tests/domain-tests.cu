#include "catch.hpp"
#include "domain.hpp"

TEST_CASE("Domain generation routines")
{
  SECTION("we should be able to allocate a Cartesian distribution")
  {
    using real = float;
    int const gl = 2;

    thrust::host_vector<point_t<real>> pts = gen_cartesian_domain<real>(gl);

    REQUIRE((pts.size() == 8));

    using point = point_t<real>;

    /* 
      Point set should be:
      0 0 0
      0 0 1
      0 1 0
      0 1 1
      1 0 0
      1 0 1
      1 1 0
      1 1 1
   */
    
    REQUIRE((pts[0] == point{0, 0, 0}));
    REQUIRE((pts[1] == point{0, 0, 1}));
    REQUIRE((pts[2] == point{0, 1, 0}));
    REQUIRE((pts[3] == point{0, 1, 1}));
    REQUIRE((pts[4] == point{1, 0, 0}));
    REQUIRE((pts[5] == point{1, 0, 1}));
    REQUIRE((pts[6] == point{1, 1, 0}));
    REQUIRE((pts[7] == point{1, 1, 1}));
  }

  SECTION("we should be able to sort by the Peanokey of each point")
  {
    using real = float;
    using point = point_t<real>;
    
    int const gl = 2;
    
    thrust::host_vector<point_t<real>> pts = gen_cartesian_domain<real>(gl);
    sort_by_peanokey<real>(pts);
    
    REQUIRE((pts[0] == point{0, 0, 0}));
    REQUIRE((pts[1] == point{0, 1, 0}));
    REQUIRE((pts[2] == point{1, 1, 0}));
    REQUIRE((pts[3] == point{1, 0, 0}));
    REQUIRE((pts[4] == point{1, 0, 1}));
    REQUIRE((pts[5] == point{1, 1, 1}));
    REQUIRE((pts[6] == point{0, 1, 1}));
    REQUIRE((pts[7] == point{0, 0, 1}));
  }

  SECTION("we should be able to sort by a device-based range")
  {
    using point = point_t<float>;
    
    thrust::device_vector<point> pts{gen_cartesian_domain<float>(2)};
    
    sort_by_peanokey<float>(pts.data(), pts.data() + pts.size());
    
    REQUIRE((pts[0] == point{0, 0, 0}));
    REQUIRE((pts[1] == point{0, 1, 0}));
    REQUIRE((pts[2] == point{1, 1, 0}));
    REQUIRE((pts[3] == point{1, 0, 0}));
    REQUIRE((pts[4] == point{1, 0, 1}));
    REQUIRE((pts[5] == point{1, 1, 1}));
    REQUIRE((pts[6] == point{0, 1, 1}));
    REQUIRE((pts[7] == point{0, 0, 1}));
  }
}
