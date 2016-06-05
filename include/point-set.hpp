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
protected:
	thrust::host_vector<T> h_x_;
	thrust::host_vector<T> h_y_;
	thrust::host_vector<T> h_z_;

	thrust::device_vector<T> d_x_;
	thrust::device_vector<T> d_y_;
	thrust::device_vector<T> d_z_;

	size_t size_ = 0;

public:
	point_set(void) = default;

	// single-parameter constructor reserves memory in
	// each vector
	point_set(size_t const size) {
		size_ = size;

		h_x_.reserve(size);
		h_y_.reserve(size);
		h_z_.reserve(size);

		d_x_.reserve(size);
		d_y_.reserve(size);
		d_z_.reserve(size);
	}

	size_t size(void) const {
		return size_;
	}
};



#endif /* POINT_SET_HPP_ */
