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

	// we should be able to traverse a point set using raw pointers
	{
		size_t const size = (1 << 8);
		point_set<float> ps{size};

		auto data = ps.get_host_ptrs();
		float* x = thrust::get<0>(data);
		float* y = thrust::get<1>(data);
		float* z = thrust::get<2>(data);

		// this may seem like a silly test but it's honestly
		// just to make sure that the interface is working correctly
		// (we can gather the pointers and write/read the data)
		for (size_t i = 0; i < size; ++i) {
			x[i] = -1;
			y[i] = -1;
			z[i] = -1;
		}

		for (size_t i = 0; i < size; ++i) {
			assert(x[i] == -1);
			assert(y[i] == -1);
			assert(z[i] == -1);
		}
	}
}
