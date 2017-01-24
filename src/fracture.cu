#include "globals.hpp"
#include "lib/fracture.hpp"
#include "math/tetra.hpp"
#include "stack-vector.hpp"

#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

__device__
int const static face_idx[4][3] = {
  { 3, 2, 1 },   // face 0
  { 0, 2, 3 },   // face 1
  { 0, 3, 1 },   // face 2
  { 0, 1, 2 } }; // face 3

__global__
void fracture_kernel(
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
    int const pa_id{pa[tid]};
    if (nm[pa_id] == 1) {
      // want to then load in the tetrahedron
      int const ta_id{ta[tid]};
      tetra const t = mesh[ta_id];
      
      // want to create something randomly accessible
      array<int, 4> t_ids;
      t_ids[0] = t.x;
      t_ids[1] = t.y;
      t_ids[2] = t.z;
      t_ids[3] = t.w;

      int const la_id{la[tid]};
      stack_vector<tetra, 4> local_tets;
      
      //printf("pa %d => ta %d => la %d\n", pa_id, ta_id, la_id);
      
      // check each bit of loc iteratively
      for (int i = 0; i < 4; ++i) {
        if (la_id & (1 << i)) {
          tetra const tmp{
            t_ids[face_idx[i][0]],
            t_ids[face_idx[i][1]],
            t_ids[face_idx[i][2]],
            pa_id};
            
          local_tets.push_back(tmp);
        }
      }
            
      // now we need to perform a write-back procedure to the main mesh
      // write back in-place
      mesh[ta_id] = local_tets[0];
      
      // now append to buffer
      int const mesh_offset{num_tetra + (tid == 0 ? 0 : fl[tid - 1])};
      for (int i = 0; i < local_tets.size() - 1; ++i) {
        mesh[mesh_offset + i] = local_tets[i + 1];
      }
    }
  }
}

using thrust::device_vector;

auto fracture(
  int const assoc_size,
  int const num_tetra,
  device_vector<int> const& pa,
  device_vector<int> const& ta,
  device_vector<int> const& la,
  device_vector<int> const& nm,
  device_vector<int> const& fl,
  device_vector<tetra>& mesh) -> void
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
    
  cudaDeviceSynchronize();
}  
