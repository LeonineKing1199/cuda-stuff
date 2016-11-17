#ifndef REGULUS_GLOBALS_HPP_
#define REGULUS_GLOBALS_HPP_

size_t const static tpb = 256; // threads per block
size_t const static bpg = 512; // blocks per grid

__device__ __inline__
auto get_tid(void) -> size_t
{
  return blockIdx.x * blockDim.x + threadIdx.x;
}

constexpr __device__
auto grid_stride(void) -> size_t
{
  return tpb * bpg;
}

#endif // REGULUS_GLOBALS_HPP_
