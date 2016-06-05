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


	cartesian_point_set(int a, int b) {

	}
};



#endif /* CARTESIAN_POINT_SET_HPP_ */
