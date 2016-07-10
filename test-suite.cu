#include <iostream>

#include "test-suite.hpp"

auto test_suite(void) -> void
{
  std::cout << "Beginning test suite!" << std::endl;

  domain_tests();

  std::cout << "Tests passed!\n" << std::endl;
}
