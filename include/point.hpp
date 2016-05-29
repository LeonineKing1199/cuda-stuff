/*
 * point.hpp
 *
 *  Created on: May 28, 2016
 *      Author: christian
 */

#ifndef POINT_HPP_
#define POINT_HPP_

#include <type_traits>

template <
	typename T,
	typename std::enable_if<std::is_floating_point<T>::value>::type>
struct point {
	T* x;
	T* y;
	T* z;
};


#endif /* POINT_HPP_ */
