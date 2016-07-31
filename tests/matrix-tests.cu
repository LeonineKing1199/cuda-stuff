#include "test-suite.hpp"
#include "../include/math/matrix.hpp"

auto matrix_tests(void) -> void
{
  std::cout << "Beginning matrix tests!" << std::endl;
  
  // we should be able to construct a matrix type
  {
    float a = 1;
    float b = 2;
    float c = 3;
    float d = 4;
    
    matrix<float, 2, 3> m{ a, b, c, d };
    
    assert((m == matrix<float, 2, 3>{ a, b, c, d }));
    assert((m != matrix<float, 2, 3>{ a, b, c, (float ) 7 }));
  }
  
  std::cout << "Passed tests!\n" << std::endl;
}