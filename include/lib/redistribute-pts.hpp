#ifndef REGULUS_LIB_REDISTRIBUTE_PTS_HPP_
#define REGULUS_LIB_REDISTRIBUTE_PTS_HPP_

#include "../globals.hpp"
#include "../math/tetra.hpp"
#include "../math/point.hpp"

template <typename T>
__global__
void redistribute_pts(
  int const assoc_size,
  int const* __restrict__ ta,
  int const* __restrict__ nm_ta,
  int const* __restrict__ nm,
  int const* __restrict__ la,
  int const* __restrict__ fl,
  tetra const* __restrict__ mesh,
  point_t<T> const* __restrict__ pts,
  int const* __restrict__ pa
  )
{
  for (auto tid = get_tid(); tid < assoc_size; tid += grid_stride()) {
    int const ta_id = ta[tid];
    
    // we encode the tid such that nm[pa[tid]] == 1 in nm_ta at ta[tid]
    int const pa_tid = nm_ta[ta_id];
    
    // this means the tetrahedron was not even written to
    if (pa_tid != -1) {
      return;
    }
    
    // this point was not ultimately nominated even though it wrote
    // this check my ultimately be unnecessary but for now I'm going
    // to be safe
    if (nm[pa_tid] == 0) {
      return;
    }
    
    // we know that we wrote to mesh at ta_id and that we then wrote
    // past the end of the mesh at fl[tid] + [0, 1, 2]
    tetra tets[4];
    int size = 0;
    
    int const fract_size = __popc(la[pa_tid]);
    int const mesh_offset = fl[pa_tid];
    
    tets[size] = mesh[ta_id];
    ++size;
    
    for (int i = 0; i < (fract_size - 1); ++i) {
      tets[size] = mesh[mesh_offset + i];
      ++size;
    }
    
    // now we can begin testing
    for (int i = 0; i < size; ++i) {
      tetra const t{ tets[i] };
      point_t<T> const a{pts[t.x]};
      point_t<T> const b{pts[t.y]};
      point_t<T> const c{pts[t.z]};
      point_t<T> const d{pts[t.w]};
      
      point_t<T> const p{pts[pa[tid]]};
      
      int const new_la{loc<T>(a, b, c, d, p)};
    }
  }
}

#endif // REGULUS_LIB_REDISTRIBUTE_PTS_HPP_
