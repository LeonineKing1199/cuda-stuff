#include "test-suite.hpp"
#include "../include/math/math.hpp"

auto math_tests(void) -> void
{
  std::cout << "Beginning math tests..." << std::endl;

  // We should be able to instantiate some of our helper types
  {
    assert(greater_than_three<4>::value == true);
    assert(greater_than_three<2>::value == false);
    
    assert(greater_than_zero<2>::value == true);
    assert(greater_than_zero<0>::value == false);
  }
    
  std::cout << "Completed math tests!\n" << std::endl;
}

