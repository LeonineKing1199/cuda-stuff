/*
 * triangulate.cu
 *
 *  Created on: Jul 3, 2016
 *      Author: christian
 */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "common.hpp"

template <typename T>
auto triangulate(thrust::host_vector<reg::point_t<T>>& h_domain) -> void
{
	// first thing we do is copy the points over to the device
	thrust::device_vector<reg::point_t<T>> d_domain{h_domain};
}


