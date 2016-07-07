/*
 * common.hpp
 *
 *  Created on: Jun 29, 2016
 *      Author: christian
 */

#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <type_traits>

namespace reg
{
	template <bool B, typename T = void>
	using enable_if_t = typename std::enable_if<B, T>::type;
}

template <
	typename T,
	typename = reg::enable_if_t<std::is_floating_point<T>::value>>
struct point_struct
{
	typedef T type;
};

template <>
struct point_struct<float>
{
	typedef float3 type;
};

template <>
struct point_struct<double>
{
	typedef double3 type;
};

namespace reg
{
	template <typename T>
	using point_t = typename point_struct<T>::type;
}

long long int peano_hilbert_key(int x, int y, int z, int bits);

#endif /* COMMON_HPP_ */
