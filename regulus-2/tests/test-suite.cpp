/*
 * test-suite.cpp
 *
 *  Created on: Jun 29, 2016
 *      Author: christian
 */

#include "test-suite.hpp"

auto test_suite(void) -> void
{
	std::cout << "Beginning test suite!" << std::endl;

	domain_tests();

	std::cout << "All tests passed!\n" << std::endl;
}


