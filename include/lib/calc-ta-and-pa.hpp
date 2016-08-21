#ifndef REGULUS_LIB_CALC_TA_AND_PA_HPP_
#define REGULUS_LIB_CALC_TA_AND_PA_HPP_

#include "../math/point.hpp"
#include "../math/tetra.hpp"
#include "../globals.hpp"

template <typename T>
__global__
void calc_initial_assoc(
  point_t<T> const* __restrict__ pts,
  int const num_pts,
  tetra const t,
  int* __restrict__ pa,
  int* __restrict__ ta,
  unsigned char* __restrict__ la)
{
  for (auto tid = get_tid(); tid < num_pts; tid += grid_stride()) {
    auto const a = pts[t.x];
    auto const b = pts[t.y];
    auto const c = pts[t.z];
    auto const d = pts[t.w];
    
    pa[tid] = tid;
    ta[tid] = 0;
    la[tid] = loc<T>(a, b, c, d, pts[tid]);
  }
}

#endif // REGULUS_LIB_CALC_TA_AND_PA_HPP_
