/*
 * point-set.hpp
 *
 *  Created on: Jun 4, 2016
 *      Author: christian
 */

#ifndef POINT_SET_HPP_
#define POINT_SET_HPP_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "globals.hpp"

/**
 * Class that acts as an interface for initializing and
 * manipulating point sets for the mesh generator
 */
template <
	typename T,
	typename = reg::enable_if_t<std::is_floating_point<T>::value>>
class point_set {
private:
	thrust::host_vector<T> h_x;
	thrust::host_vector<T> h_y;
	thrust::host_vector<T> h_z;

	thrust::device_vector<T> d_x;
	thrust::device_vector<T> d_y;
	thrust::device_vector<T> d_z;

public:
	point_set(void) = default;

	// single-parameter constructor reserves memory in
	// each vector
	point_set(size_t const size) {
		h_x.reserve(size);
	}
};



#endif /* POINT_SET_HPP_ */
