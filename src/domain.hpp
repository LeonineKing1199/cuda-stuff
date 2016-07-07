/*
 * domain.cu
 *
 *  Created on: Jun 29, 2016
 *      Author: christian
 *
 * Routines used to generate the point set to triangulate
 */

#ifndef DOMAIN_HPP_
#define DOMAIN_HPP_

#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include "common.hpp"

/**
 * Function that returns a Cartesian domain with point coordinates ranging
 * from (0, 0, 0) to (gl - 1, gl - 1, gl - 1) for a total of gl ^ 3 points
 */
template <typename T>
auto gen_cartesian_domain(int const gl) -> thrust::host_vector<reg::point_t<T>>
{
	int const num_points = gl * gl * gl;
	thrust::host_vector<typename point_struct<T>::type> point_set;

	point_set.reserve(num_points);

	for (int x = 0; x < gl; ++x)
		for (int y = 0; y < gl; ++y)
			for (int z = 0; z < gl; ++z)
				point_set.push_back(
						reg::point_t<T>{
							static_cast<T>(x),
							static_cast<T>(y),
							static_cast<T>(z)
						});

	return point_set;
}

/**
 * Function that takes a host_vector of reg::point_t<T> and converts
 * the point coordinates to integers and sorts them according to the
 * Peano-Hilbert curve.
 *
 * A unary functor is required for the T to integer conversion.
 *
 * This probably isn't the most efficient code and isn't device-optimized right now.
 * A device version may be implemented later.
 * And a hard-code conversion routine might be as well.
 */
template <typename RandomAccessIterator>
auto sort_domain_by_peanokey(
		RandomAccessIterator first,
		RandomAccessIterator last)
-> void
{
	using point_t = typename RandomAccessIterator::value_type;
	using peanokey = long long int;

	struct peano_hash : public thrust::unary_function<point_t, peanokey>
	{
		__host__ __device__
		peanokey operator()(point_t p) const
		{
			return peano_hilbert_key(
					static_cast<int>(p.x),
					static_cast<int>(p.y),
					static_cast<int>(p.z),
					23);
		}
	};

	thrust::host_vector<peanokey> keys{
		thrust::make_transform_iterator(first, peano_hash{}),
		thrust::make_transform_iterator(last, peano_hash{})};

	thrust::sort_by_key(keys.begin(), keys.end(), first);
}

#endif /* DOMAIN_HPP_ */
