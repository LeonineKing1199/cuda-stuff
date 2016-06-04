/*
 * math.hpp
 *
 *  Created on: May 29, 2016
 *      Author: christian
 */

#ifndef MATH_HPP_
#define MATH_HPP_

#include "../include/globals.hpp"

template <
	typename T,
	typename = reg::enable_if_t<std::is_floating_point<T>::value>>
__host__ __device__
T det3(T const* m) { // size = 9
	T const a = m[0] * m[4] * m[8];
	T const b = m[1] * m[5] * m[6];
	T const c = m[2] * m[3] * m[7];
	T const d = m[2] * m[4] * m[6];
	T const e = m[1] * m[3] * m[8];
	T const f = m[0] * m[5] * m[7];

	return (a + b + c - d - e - f);
}

template <
	typename T,
	typename = reg::enable_if_t<std::is_floating_point<T>::value>>
__host__ __device__
T det4(T const* m) { // size = 16
	T const a = m[0 * 4 + 0];
	T const b = m[0 * 4 + 1];
	T const c = m[0 * 4 + 2];
	T const d = m[0 * 4 + 3];

	T const A[9] = { m[5], m[6], m[7], m[9], m[10], m[11], m[13], m[14], m[15] };
	T const B[9] = { m[4], m[6], m[7], m[8], m[10], m[11], m[12], m[14], m[15] };
	T const C[9] = { m[4], m[5], m[7], m[8], m[9], m[11], m[12], m[13], m[15] };
	T const D[9] = { m[4], m[5], m[6], m[8], m[9], m[10], m[12], m[13], m[14] };

	return ((a * det3(A)) - (b * det3(B)) + (c * det3(C)) - (d * det3(D)));
}




#endif /* MATH_HPP_ */
