#ifndef REGULUS_LIB_CALC_TA_AND_PA_HPP_
#define REGULUS_LIB_CALC_TA_AND_PA_HPP_

#include "globals.hpp"
#include "index_t.hpp"
#include "math/point.hpp"
#include "math/tetra.hpp"

template <typename T>
__global__
void calc_initial_assoc(
  point_t<T> const* __restrict__ pts,
  int const num_pts,
  tetra const t,
  index_t* __restrict__ pa,
  index_t* __restrict__ ta,
  loc_t* __restrict__ la)
{
  for (auto tid = get_tid(); tid < num_pts; tid += grid_stride()) {
    auto const a = pts[t.x];
    auto const b = pts[t.y];
    auto const c = pts[t.z];
    auto const d = pts[t.w];
    
    pa[tid] = index_t{static_cast<long long>(tid)};
    ta[tid] = index_t{0ll};
    la[tid] = loc<T>(a, b, c, d, pts[tid]);
  }
}

#endif // REGULUS_LIB_CALC_TA_AND_PA_HPP_
