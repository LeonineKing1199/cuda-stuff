#include "test-suite.hpp"

auto test_suite(void) -> void
{
  std::cout << "Beginning test suite...\n" << std::endl;
  
  array_tests();
  stack_vector_tests();
  domain_tests();
  math_tests();
  matrix_tests();
  tetra_tests();
  nomination_tests();
  fract_location_tests();
  fracture_tests();
  get_assoc_size_tests();
  redistribute_pts_tests();
  mesher_tests();
  
  std::cout << "Tests passed!\n" << std::endl;
}