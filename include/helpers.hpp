/*
 * helpers.hpp
 *
 *  Created on: May 28, 2016
 *      Author: christian
 */

#ifndef HELPERS_HPP_
#define HELPERS_HPP_

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


#endif /* HELPERS_HPP_ */
