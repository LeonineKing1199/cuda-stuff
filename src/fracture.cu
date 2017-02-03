#include "globals.hpp"
#include "lib/fracture.hpp"

#include <stdio.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <thrust/distance.h>

namespace T = thrust;

__device__
size_t const static face_idx[4][3] = {
  { 3, 2, 1 },   // face 0
  { 0, 2, 3 },   // face 1
  { 0, 3, 1 },   // face 2
  { 0, 1, 2 }    // face 3
};

__global__
void fracture_kernel(
  size_t const assoc_size,
  size_t const num_tetra,
  index_t  const* __restrict__ pa,
  index_t  const* __restrict__ ta,
  loc_t    const* __restrict__ la,
  unsigned const* __restrict__ nm,
  index_t  const* __restrict__ fl,
  tetra         * __restrict__ mesh)
{
  for (auto tid = get_tid(); tid < assoc_size; tid += grid_stride()) {
    // if this point was nominated...
    index_t const pa_id = pa[tid];
    if (nm[pa_id] == 1) {

      // want to then load in the tetrahedron
      index_t const ta_id = ta[tid];
      tetra   const t     = mesh[ta_id];
      
      // want to create something randomly accessible
      array<int, 4> vertex_idx = { t.x, t.y, t.z, t.w };

      loc_t const loc = la[tid];
    
      //printf("pa %d => ta %d => la %d\n", pa_id, ta_id, loc);
      
      array<tetra, 4>   fract_set; 
      array<index_t, 4> fract_locs;

      // check each bit of loc iteratively
      for (decltype(fract_set.size()) i = 0; i < fract_set.size(); ++i) {
        if (loc & (1 << i)) {
          fract_set[i] = { 
            vertex_idx[face_idx[i][0]],
            vertex_idx[face_idx[i][1]],
            vertex_idx[face_idx[i][2]],
            pa_id
          };

          fract_locs[i] = 

        } else {
          fract_set[i]  = {-1, -1, -1, -1};
          fract_locs[i] = -1;
        }
      }

      // auto const end = T::remove_copy_if(
      //   T::seq, 
      //   fract_set.begin(), fract_set.end(),
      //   filtered_fract_set.begin(),
      //   [](tetra const& t) -> bool { return t.x == -1; });

      // // perform the first write-back to the location where the
      // // fractured tetrahedron was stored
      // auto begin = filtered_fract_set.begin();

      // mesh[ta_id] = *begin;
      // ++begin;

      // size_t mesh_offset = num_tetra + (tid == 0 ? 0 : fl[tid - 1]);

      // for (auto it = begin; it < end; ++it) {
      //   mesh[mesh_offset++] = *it;
      // }
    }
  }
}

auto fracture(
  size_t const assoc_size,
  size_t const num_tetra,
  T::device_vector<index_t>  const& pa,
  T::device_vector<index_t>  const& ta,
  T::device_vector<loc_t>    const& la,
  T::device_vector<unsigned> const& nm,
  T::device_vector<index_t>  const& fl,
  T::device_vector<tetra>         & mesh) -> void
{
  fracture_kernel<<<bpg, tpb>>>(
    assoc_size,
    num_tetra,
    pa.data().get(),
    ta.data().get(),
    la.data().get(),
    nm.data().get(),
    fl.data().get(),
    mesh.data().get());
}  
