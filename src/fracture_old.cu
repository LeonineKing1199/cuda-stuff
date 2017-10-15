#include "globals.hpp"
#include "lib/fracture.hpp"
#include "lib/set-internal-fract-adjacencies.hpp"

#include <stdio.h>

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
  tetra         * __restrict__ mesh,
  adjacency     * __restrict__ adjacency_relations)
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
      
      array<index_t, 4> fract_locs;
      
      size_t mesh_offset = num_tetra + (tid == 0 ? 0 : fl[tid - 1]);

      bool first_fracture = true;
      // check each bit of loc iteratively
      for (decltype(fract_locs.size()) i = 0; i < fract_locs.size(); ++i) {
        // if the bit is set...
        if (loc & (1 << i)) {
          // make a tetrahedron out of face i 
          // and the insertion point
          auto const fract_loc = first_fracture ? ta_id : mesh_offset++;
          fract_locs[i]  = fract_loc;
          first_fracture = false;

          mesh[fract_loc] = tetra{
            vertex_idx[face_idx[i][0]],
            vertex_idx[face_idx[i][1]],
            vertex_idx[face_idx[i][2]],
            pa_id};

        } else {
          fract_locs[i] = -1;
        }
      }

      array<adjacency, 4> const adj_relations = 
        set_interal_fract_adjacencies(fract_locs, loc);

      for (decltype(fract_locs.size()) i = 0; i < fract_locs.size(); ++i) {
        auto const fract_loc = fract_locs[i];

        adjacency_relations[fract_loc] = (
          static_cast<bool>(fract_loc)
          ? adj_relations[i]
          : adjacency{ -1, -1, -1, -1 });
      }
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
  T::device_vector<tetra>         & mesh,
  T::device_vector<adjacency>     & adjacency_relations) -> void
{
  fracture_kernel<<<bpg, tpb>>>(
    assoc_size,
    num_tetra,
    pa.data().get(),
    ta.data().get(),
    la.data().get(),
    nm.data().get(),
    fl.data().get(),
    mesh.data().get(),
    adjacency_relations.data().get());
}  
