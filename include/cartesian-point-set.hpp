/*
 * cartesian-point-set.hpp
 *
 *  Created on: Jun 4, 2016
 *      Author: christian
 */

#ifndef CARTESIAN_POINT_SET_HPP_
#define CARTESIAN_POINT_SET_HPP_

#include "point-set.hpp"

template <
	typename T,
	typename = reg::enable_if_t<std::is_floating_point<T>::value>>
class cartesian_point_set : private point_set<T> {
public:
	// inherit constructors from our base class
	using point_set<T>::point_set;

	cartesian_point_set(void) = default;

	cartesian_point_set(int const a, int const b) {
		int const grid_length{b - a};

		point_set<T>::size_ = grid_length * grid_length * grid_length;
		point_set<T>::resize(point_set<T>::size_);

		for (decltype(point_set<T>::size_) i = 0; i < point_set<T>::size_; ++i) {
			unsigned int const x = i / (grid_length * grid_length);
			unsigned int const y = (i - x * (grid_length * grid_length)) / grid_length;
			unsigned int const z = i - x * (grid_length * grid_length) - y * grid_length;

			point_set<T>::h_x_[i] = x;
			point_set<T>::h_y_[i] = y;
			point_set<T>::h_z_[i] = z;
		}

		point_set<T>::copy_host_to_device();
	}

	size_t size(void) const {
		return point_set<T>::size_;
	}
};



#endif /* CARTESIAN_POINT_SET_HPP_ */
