#ifndef REGULUS_GLOBALS_HPP_
#define REGULUS_GLOBALS_HPP_

#include <cstddef>

namespace regulus
{
  constexpr std::size_t const static tpb = 256; // threads per block
  constexpr std::size_t const static bpg = 512; // blocks per grid

  // tid = thread id
  __device__ __inline__
  auto get_tid(void) -> std::size_t
  {
    return blockIdx.x * blockDim.x + threadIdx.x;
  }

  constexpr __device__
  auto grid_stride(void) -> std::size_t
  {
    return tpb * bpg;
  }
}

#endif // REGULUS_GLOBALS_HPP_
