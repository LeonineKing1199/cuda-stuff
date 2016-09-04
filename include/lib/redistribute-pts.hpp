#ifndef REGULUS_LIB_REDISTRIBUTE_PTS_HPP_
#define REGULUS_LIB_REDISTRIBUTE_PTS_HPP_

#include "../globals.hpp"
#include "../math/tetra.hpp"
#include "../math/point.hpp"
#include "../array.hpp"

template <typename T>
__global__
void redistribute_pts(
  int const assoc_size,
  tetra const* __restrict__ mesh,
  point_t<T> const* __restrict__ pts,
  int const* __restrict__ nm,
  int const* __restrict__ nm_ta,
  int const* __restrict__ fl,
  int* pa,
  int* ta,
  int* la,
  int* num_redistributions)
{
  for (auto tid = get_tid(); tid < assoc_size; tid += grid_stride()) {
    // store a copy of the current ta value
    int const ta_id = ta[tid];
    int const pa_id = pa[tid];
    
    // we encode a tid such that nm[pa[tid]] == 1 in nm_ta at ta[tid]
    // i.e., if (nm[pa[tid]] == 1) then nm_ta[ta[tid]] == tid
    // and we want to test validity of nm_ta[ta[tid]] because we're going
    // in reverse
    int const pa_tid = nm_ta[ta_id];
    
    // this means the tetrahedron was not even written to
    if (pa_tid != -1) {
      return;
    }
    
    // this point was not ultimately nominated even though it wrote
    // this check my ultimately be unnecessary but for now I'm going
    // to be safe (I think it's necessary)
    if (nm[pa_tid] != 1) {
      return;
    }
    
    // we now know that pa_tid is actually a valid tuple id!
    // invalidate this association
    ta[tid] = -1;
    pa[tid] = -1;
    la[tid] = -1;
    
    // we know that we wrote to mesh at ta_id and that we then wrote
    // past the end of the mesh at fl[tid] + { 0[, 1[, 2]] }
    array<int, 4> local_pa{-1};
    array<int, 4> local_ta{-1};
    array<int, 4> local_la{-1};
    
    int la_size = 0;
    
    array<tetra, 4> tets;
    int tet_size = 0;
    
    // load the tetrahedra onto the stack
    tets[tet_size] = mesh[ta_id];
    local_ta[tet_size] = ta_id;
    local_pa[tet_size] = pa_id;
    ++tet_size;
    
    int const fract_size = __popc(la[pa_tid]);
    int const mesh_offset = fl[pa_tid];
    for (int i = 0; i < (fract_size - 1); ++i) {
      tets[tet_size] = mesh[mesh_offset + i];
      local_ta[tet_size] = mesh_offset + i;
      local_pa[tet_size] = pa_id;
      ++tet_size;
    }
    
    // now we can begin testing each one
    for (int i = 0; i < tet_size; ++i) {
      tetra const t{tets[i]};
      
      point_t<T> const a{pts[t.x]};
      point_t<T> const b{pts[t.y]};
      point_t<T> const c{pts[t.z]};
      point_t<T> const d{pts[t.w]};
      
      point_t<T> const p{pts[pa_id]};
      
      local_la[la_size] = loc<T>(a, b, c, d, p);
      ++la_size;
    }
    
    // now we have to do a write-back to main memory
    int const assoc_offset = 4 * atomicAdd(num_redistributions, 1);
    
    *reinterpret_cast<int4*>(pa) = *reinterpret_cast<int4*>(local_pa.begin());
    *reinterpret_cast<int4*>(ta) = *reinterpret_cast<int4*>(local_ta.begin());
    *reinterpret_cast<int4*>(la) = *reinterpret_cast<int4*>(local_la.begin());
  }
}

#endif // REGULUS_LIB_REDISTRIBUTE_PTS_HPP_
