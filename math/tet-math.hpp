/*
 * tet-math.hpp
 *
 *  Created on: Jun 12, 2016
 *      Author: christian
 */

#ifndef TET_MATH_HPP_
#define TET_MATH_HPP_

#include <array>

#include "../include/globals.hpp"

/**
 * Function that returns orientation of the current tetrahedron
 *
 * We take tet to be a 12 element array and pt to be a 3 element array
 */
template <
	typename T,
	typename = reg::enable_if_t<std::is_floating_point<T>::value>>
__host__ __device__
T ort(std::array<T, 12> tet, std::array<T, 3> pt) {

	return T{};
}


#endif /* TET_MATH_HPP_ */
