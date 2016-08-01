#include "test-suite.hpp"

auto test_suite(void) -> void
{
  std::cout << "Beginning test suite...\n" << std::endl;
  
  domain_tests();
  math_tests();
  mesher_tests();
  array_tests();
  matrix_tests();
  
  std::cout << "Tests passed!\n" << std::endl;
}