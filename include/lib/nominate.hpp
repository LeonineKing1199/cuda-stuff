#ifndef REGULUS_LIB_NOMINATE_HPP_
#define REGULUS_LIB_NOMINATE_HPP_

#include "../math/point.hpp"
#include "../math/tetra.hpp"
#include "../globals.hpp"

template <typename T>
__global__
void nominate(
  int const assoc_size,
  int const* __restrict__ ta,
  int const* __restrict__ pa,
  int* nm_ta,
  int* nm)
{
  for (auto tid = get_tid(); tid < assoc_size; tid += grid_stride()) {
    int* address = nm_ta + ta[tid];
    int compare = 0;
    int val = 1;
    
    // this thread was the first one to find this tetrahedron
    if (atomicCAS(address, compare, val) == 0) {
      if (atomicOr(nm + pa[tid], 1) == 0) {
        atomicAnd(nm + pa[tid], 0);
      }
    } else {
      atomicAnd(nm + pa[tid], 0);
    }
  }
}

#endif // REGULUS_LIB_NOMINATE_HPP_
