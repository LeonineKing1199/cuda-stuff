#ifndef REGULUS_GLOBALS_HPP_
#define REGULUS_GLOBALS_HPP_

int const static tpb = 256; // threads per block
int const static bpg = 512; // blocks per grid

__device__
auto get_tid(void) -> int
{
  return blockIdx.x * blockDim.x + threadIdx.x;
}

constexpr
__device__
auto grid_stride(void) -> int
{
  return tpb * bpg;
}

#endif // REGULUS_GLOBALS_HPP_
