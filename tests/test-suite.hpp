#ifndef REGULUS_TEST_SUITE_HPP_
#define REGULUS_TEST_SUITE_HPP_

#include <iostream>
#include <cassert>

#include "../include/math/point.hpp"

auto test_suite(void) -> void;
auto domain_tests(void) -> void;
auto mesher_tests(void) -> void;
auto math_tests(void) -> void;
auto array_tests(void) -> void;
auto matrix_tests(void) -> void;
auto tetra_tests(void) -> void;
auto nomination_tests(void) -> void;
auto fract_location_tests(void) -> void;
auto fracture_tests(void) -> void;
auto redistribute_pts_tests(void) -> void;
auto stack_vector_tests(void) -> void;
auto get_assoc_size_tests(void) -> void;

#endif // REGULUS_TEST_SUITE_HPP_

