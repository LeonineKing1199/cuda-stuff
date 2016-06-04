#include "../include/globals.hpp"

__device__
unsigned int get_tid(void) {
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__
unsigned int get_stride(void) {
	return blockDim.x * gridDim.x;
}
