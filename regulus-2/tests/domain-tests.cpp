/*
 * domain-tests.cu
 *
 *  Created on: Jun 29, 2016
 *      Author: christian
 */

#include "test-suite.hpp"
#include "../src/domain.hpp"

auto domain_tests(void) -> void
{
	typedef float real;
	using thrust::system::detail::generic::select_system;

	int const grid_length = 128;

	/*
	 * We should be able to create a basic Cartesian distribution of points
	 */
	thrust::host_vector<reg::point_t<real>> domain = gen_cartesian_domain<real>(grid_length);

	assert(domain.size() == grid_length * grid_length * grid_length);

	auto& first_point = domain.front();
	auto& last_point = domain.back();

	assert(first_point.x == 0);
	assert(first_point.y == 0);
	assert(first_point.z == 0);

	assert(last_point.x == grid_length - 1);
	assert(last_point.y == grid_length - 1);
	assert(last_point.z == grid_length - 1);

	/*
	 * We should be able to sort it according to the Peano-Hilbert curve
	 */
	sort_domain_by_peanokey(domain.begin(), domain.end());
}

