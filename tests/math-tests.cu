#include "test-suite.hpp"
#include "../include/math.hpp"

auto math_tests(void) -> void
{
  std::cout << "Beginning math tests..." << std::endl;
  
  // We should be able to take the determinant of a 3x3
  {
    matrix_t<float, 3, 3> m{
      9, 0, 0,
      0, 9, 0,
      0, 0, 9};
    
    assert((m.size() == 9));
    
    assert(det(m) == 729);
    
    matrix_t<float, 3, 3> other_m{
      35, 99, 93,
      2, 16, 80,
      14, 4, 48};
      
    assert(det(other_m) == 96968.000);
  }
  
  std::cout << "Completed math tests!\n" << std::endl;
}

