#include <thrust/device_vector.h>
#include <vector>

#include "test-suite.hpp"
#include "../include/lib/fract-locations.hpp"

auto fract_location_tests(void) -> void
{
  std::cout << "Beginning fracture location tests..." << std::endl;
  
  // We should be able to accurately determine fracture indices to write to
  {
    std::vector<int> const h_pa{0, 0, 0, 1, 2, 2, 3, 3, 3};
    thrust::device_vector<int> pa{h_pa};
    
    std::vector<int> const h_la{3, 3, 3, 15, 7, 7, 3, 3, 3};
    thrust::device_vector<int> la{h_la};
    
    std::vector<int> const h_nm{1, 1, 0, 1};
    thrust::device_vector<int> nm{h_nm};
    
    thrust::device_vector<int> fl{ pa.size(), -1 };
    
    fract_locations(
      pa.data().get(),
      nm.data().get(),
      la.data().get(),
      pa.size(),
      fl.data().get());
      
    std::vector<int> expected_vals{0, 1, 2, 3, 6, 6, 6, 7, 8};
    
    assert(fl.size() == expected_vals.size());
    
    for (unsigned int i = 0; i < expected_vals.size(); ++i) {
      assert(fl[i] == expected_vals[i]);
    }
  }
  
  std::cout << "Tests passed!\n" << std::endl;
}