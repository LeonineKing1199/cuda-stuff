#include "test-suite.hpp"

auto test_suite(void) -> void
{
  std::cout << "Beginning test suite...\n" << std::endl;
  
  array_tests();
  domain_tests();
  math_tests();
  matrix_tests();
  tetra_tests();
  mesher_tests();
  
  std::cout << "Tests passed!\n" << std::endl;
}