/*
 * math.hpp
 *
 *  Created on: May 29, 2016
 *      Author: christian
 */

#ifndef MATH_HPP_
#define MATH_HPP_

#include <type_traits>

template <
	typename T,
	typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
__host__ __device__
T det3(T* m) {
	T const a = m[0] * m[4] * m[8];
	T const b = m[1] * m[5] * m[6];
	T const c = m[2] * m[3] * m[7];
	T const d = m[2] * m[4] * m[6];
	T const e = m[1] * m[3] * m[8];
	T const f = m[0] * m[5] * m[7];

	return (a + b + c - d - e - f);
}




#endif /* MATH_HPP_ */
