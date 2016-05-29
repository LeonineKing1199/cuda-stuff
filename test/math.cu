#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cassert>
#include <iostream>

#include "../math/math.hpp"
#include "test.hpp"
#include "../include/helpers.hpp"

__global__
void calc_det3(float const* matrices, unsigned int const num_matrices, float* dets) {
	for (auto tid = get_tid(); tid < num_matrices; tid += get_stride()) {
		float vals[9] = { 0 };

		for (int i = 0; i < 9; ++i) {
			vals[i] = matrices[i * num_matrices + tid];
		}

		dets[tid] = det3<float>(vals);
	}
}

void determinant_tests(void) {
	// we use some number of matrices...
	unsigned int const num_matrices = (1 << 8);

	// we use 9 randomly generated values
	// the expected determinant in this case is: -11263739881452.000
	float const matrix_values[9] = { 87432, 84228, 93841, 65001, 51520, 65406, 23748, 48720, 47247 };

	thrust::device_vector<float> device_matrices{9 * num_matrices};
	thrust::host_vector<float> host_floats{9 * num_matrices};

	thrust::device_vector<float> device_dets{num_matrices};
	thrust::host_vector<float> host_dets{num_matrices};

	// for the sake of reliable testing, each matrix is the same
	for (int i = 0; i < 9; ++i) { // 9 rows
		for (int j = 0; j < num_matrices; ++j) {
			host_floats[i * num_matrices + j] = matrix_values[i];
		}
	}

	device_matrices = host_floats;

	calc_det3<<<bpg, tpb>>>(device_matrices.data().get(), num_matrices, device_dets.data().get());
	cudaDeviceSynchronize();

	host_dets = device_dets;

	for (auto &v : host_dets) {
		assert(static_cast<long int>(v) == -11263705874432);
	}
}

