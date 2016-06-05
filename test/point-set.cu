#include "test.hpp"
#include "../include/point-set.hpp"
#include "../include/cartesian-point-set.hpp"

void point_set_tests(void) {
	// it should be default-constructible
	{
		point_set<float> ps;
	}

	// it should be able to reserve initial memory
	{
		size_t const size = (1 << 8);
		point_set<float> ps{size};
	}

	// we should have a Cartesian point set implementation
	{
		//size_t const size = (1 << 8);
		cartesian_point_set<float> cps;
	}
}
