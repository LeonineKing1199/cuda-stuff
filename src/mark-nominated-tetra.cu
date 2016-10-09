#include "../include/lib/mark-nominated-tetra.hpp"
#include "../include/globals.hpp"

__global__
void mark_nominated_tetra(
  int const assoc_size,
  int const* __restrict__ pa,
  int const* __restrict__ ta,
  int const* __restrict__ nm,
  int* __restrict__ nm_ta)
{
  for (auto tid = get_tid(); tid < assoc_size; tid += grid_stride()) {
    if (nm[pa[tid]] == 1) {
      nm_ta[ta[tid]] = tid;
    }
  }
}