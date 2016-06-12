/*
 * helpers.hpp
 *
 *  Created on: May 28, 2016
 *      Author: christian
 */

#ifndef GLOBALS_HPP_
#define GLOBALS_HPP_

#include <thrust/tuple.h>
#include <type_traits>

// blocks per grid
int const static bpg = 512;

// threads per block
int const static tpb = 128;

// gets the currently executing thread's id
__device__
unsigned int get_tid(void);

// gets the size of the current grid stride
__device__
unsigned int get_stride(void);

namespace reg {
	template <bool B, typename T = void >
	using enable_if_t = typename std::enable_if<B, T>::type;

	template <
		typename T,
		typename = enable_if_t<std::is_floating_point<T>::value>>
	using point_tuple = typename thrust::tuple<T*, T*, T*>;
}

// some convenience typedefs for easier refactoring in the future
typedef unsigned int integral;
typedef uint4 integral4;

#endif /* GLOBALS_HPP_ */
