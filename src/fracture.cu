#include <stdio.h>

#include "../include/lib/fracture.hpp"

__device__ int const static face_idx[4][3] = {
  { 3, 2, 1 }, // face 0
  { 0, 2, 3 }, // face 1
  { 0, 3, 1 }, // face 2
  { 0, 1, 2 }  // face 3
};

__global__
void fracture(
  int const assoc_size,
  int const num_tetra,
  int const* __restrict__ pa,
  int const* __restrict__ ta,
  int const* __restrict__ la,
  int const* __restrict__ nm,
  int const* __restrict__ fl,
  tetra* __restrict__ mesh)
{
  for (auto tid = get_tid(); tid < assoc_size; tid += grid_stride()) {
    // if this point was nominated...
    int const pa_id = pa[tid];
    if (nm[pa_id] == 1) {
      // want to then load in the tetrahedron
      int const ta_id = ta[tid];
      tetra const t = mesh[ta_id];
      array<int, 4> t_ids;
      t_ids[0] = t.x;
      t_ids[1] = t.y;
      t_ids[2] = t.z;
      t_ids[3] = t.w;
      
      int const loc{la[tid]};

      array<tetra, 4> local_tets;
      int tets_size = 0;
      
      // check each bit of loc iteratively
      for (int i = 0; i < 4; ++i) {
        if (loc & (1 << i)) {          
          local_tets[tets_size] = tetra{
            t_ids[face_idx[i][0]],
            t_ids[face_idx[i][1]],
            t_ids[face_idx[i][2]],
            pa_id};
          ++tets_size;
        }
      }
      
      // now we need to perform a write-back procedure to the main mesh
      int const mesh_offset = num_tetra + fl[tid];
      int const adjusted_fract_size = __popc(loc) - 1;
      
      mesh[ta_id] = local_tets[0];
      
      for (int i = 0; i < adjusted_fract_size; ++i) {
        mesh[mesh_offset + i] = local_tets[i + 1];
      }
    }
  }
}