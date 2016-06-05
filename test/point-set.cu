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
		assert(ps.size() == size);
	}

	// we should have a Cartesian point set implementation
	{
		cartesian_point_set<float> cps;
		assert(cps.size() == 0);
	}

	// we should be able to construct a basic Cartesian grid
	{
		cartesian_point_set<float> cps{0, 9};
		assert(cps.size() == (9 * 9 * 9));
	}
}
