#include "test-suite.hpp"
#include "../include/domain.hpp"
#include "../include/timer.hpp"

auto domain_tests(void) -> void
{
  std::cout << "Running domain-tests" << std::endl;

  // We should be able to declare a point type
  {
    typename point_type<float>::type point{0, 1, 2};

    assert(point.x == 0);
    assert(point.y == 1);
    assert(point.z == 2);
  }

  {
    typename point_type<double>::type point{0, 1, 2};

    assert(point.x == 0);
    assert(point.y == 1);
    assert(point.z == 2);
  }

  // We should be able to allocate a Cartesian distribution
  {
    using real = float;
    int const gl = 2;

    thrust::host_vector<point_t<real>> pts = gen_cartesian_domain<real>(gl);

    assert(pts.size() == 8);

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
    
    assert((pts[0] == point{0, 0, 0}));
    assert((pts[1] == point{0, 0, 1}));
    assert((pts[2] == point{0, 1, 0}));
    assert((pts[3] == point{0, 1, 1}));
    assert((pts[4] == point{1, 0, 0}));
    assert((pts[5] == point{1, 0, 1}));
    assert((pts[6] == point{1, 1, 0}));
    assert((pts[7] == point{1, 1, 1}));
  }
  
  // We should be able to sort by the Peanokey of each point (device version)
  {
    using real = float;
    
    timer t;
    t.start();
    
    int const gl = 12;
    
    thrust::host_vector<point_t<real>> pts{gen_cartesian_domain<real>(gl)};
    sort_by_peanokey<real>(pts);
    
    t.end();
    //std::cout << "Device spatial sorting completed in : " << t.get_duration() << " seconds" << std::endl;
  }
  
  // We should be able to sort by the Peanokey of each point (host version)
  {
    using real = float;
    using point = point_t<real>;
    
    int const gl = 2;
    
    thrust::host_vector<point_t<real>> pts = gen_cartesian_domain<real>(gl);
    sort_by_peanokey<real>(pts);
    
    assert((pts[0] == point{0, 0, 0}));
    assert((pts[1] == point{0, 1, 0}));
    assert((pts[2] == point{1, 1, 0}));
    assert((pts[3] == point{1, 0, 0}));
    assert((pts[4] == point{1, 0, 1}));
    assert((pts[5] == point{1, 1, 1}));
    assert((pts[6] == point{0, 1, 1}));
    assert((pts[7] == point{0, 0, 1}));
  }
  
  std::cout << "Completed domain tests!\n" << std::endl;
}
