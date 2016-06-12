/*
 * tet-math.hpp
 *
 *  Created on: Jun 12, 2016
 *      Author: christian
 */

#ifndef TET_MATH_HPP_
#define TET_MATH_HPP_

#include "../include/globals.hpp"

/**
 * Function that returns orientation of the current tetrahedron
 */
template <
	typename T,
	typename = reg::enable_if_t<std::is_floating_point<T>::value>>
T ort(integral4 tet, reg::point_tuple<T> pts) {
	T* x = thrust::get<0>(pts);
	T* y = thrust::get<1>(pts);
	T* z = thrust::get<2>(pts);



	return T{};
}


#endif /* TET_MATH_HPP_ */
