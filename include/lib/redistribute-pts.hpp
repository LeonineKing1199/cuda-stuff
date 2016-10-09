#ifndef REGULUS_LIB_REDISTRIBUTE_PTS_HPP_
#define REGULUS_LIB_REDISTRIBUTE_PTS_HPP_

#include <cstdio>
#include <cassert>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "../globals.hpp"
#include "../math/tetra.hpp"
#include "../math/point.hpp"
#include "../array.hpp"
#include "../stack-vector.hpp"
#include "mark-nominated-tetra.hpp"

using thrust::copy;
using thrust::device_vector;

template <typename T>
__global__
void redistribute_pts_kernel(
  int const assoc_size,
  int const num_tetra,
  tetra const* __restrict__ mesh,
  point_t<T> const* __restrict__ pts,
  int const* nm,
  int const* __restrict__ nm_ta,
  int const* __restrict__ fl,
  int* pa,
  int* ta,
  int* la,
  int* num_redistributions)
{
  for (auto tid = get_tid(); tid < assoc_size; tid += grid_stride()) {
    // store a copy of the current ta value
    int const ta_id{ta[tid]};
    int const pa_id{pa[tid]};
    
    // we store a mapping between the index of a
    // tetrahedron being fractured and the index
    // of the the association arrays
    int const tuple_id = nm_ta[ta_id];

    // this means the tetrahedron was not even written to
    if (tuple_id == -1) {
      return;
    }

    // if the point was not nominated, return
    if (nm[pa[tuple_id]] != 1) {
      return;
    }

    // if this thread is the current thread, we should NOT
    // invalidate the association so we should just bail
    if (tuple_id == tid) {
      return;
    }

    // we now know that tuple_id is actually a valid tuple id!
    // invalidate this association
    ta[tid] = -1;
    pa[tid] = -1;
    la[tid] = -1;

    // we know that we wrote to mesh at ta_id and that we then wrote
    // past the end of the mesh at fl[tuple_id] + { 0[, 1[, 2]] }
    stack_vector<int, 4> local_pa;
    stack_vector<int, 4> local_ta;
    stack_vector<int, 4> local_la;
    
    stack_vector<tetra, 4> tets;

    // load the tetrahedra onto the stack
    tets.push_back(mesh[ta_id]);
    local_ta.push_back(ta_id);
    local_pa.push_back(pa_id);
    
    int const fract_size{__popc(la[tuple_id])};
    int const mesh_offset{num_tetra + (tuple_id == 0 ? 0 : fl[tuple_id - 1])};
    
    for (int i = 0; i < (fract_size - 1); ++i) {
      int const tet_idx = mesh_offset + i;
      tets.push_back(mesh[tet_idx]);
      local_ta.push_back(tet_idx);
      local_pa.push_back(pa_id);
    }

    // now we can begin testing each one
    point_t<T> const p = pts[pa_id];
    for (int i = 0; i < tets.size(); ++i) {
      tetra const t = tets[i];
      
      point_t<T> const a = pts[t.x];
      point_t<T> const b = pts[t.y];
      point_t<T> const c = pts[t.z];
      point_t<T> const d = pts[t.w];
      
      local_la.push_back(loc<T>(a, b, c, d, p));
    }

    // this is some manual clean-up
    // if the location code is -1, we should just
    // null this assocation out completely
    for (int i = 0; i < local_la.size(); ++i) {
      if (local_la[i] == -1) {
        local_pa[i] = -1;
        local_ta[i] = -1;
      }
    }
    
    for (int i = local_la.size(); i < 4; ++i) {
      local_la.push_back(-1);
      local_pa.push_back(-1);
      local_ta.push_back(-1);
    }

    /*printf("New association tuple:\npa: %d %d %d %d\nta: %d %d %d %d\nla: %d %d %d %d\n\n",
      local_pa[0], local_pa[1], local_pa[2], local_pa[3],
      local_ta[0], local_ta[1], local_ta[2], local_ta[3],
      local_la[0], local_la[1], local_la[2], local_la[3]);//*/

    // now we have to do a write-back to main memory
    int const assoc_offset{assoc_size + (4 * atomicAdd(num_redistributions, 1))};
        
    copy(thrust::seq, local_pa.begin(), local_pa.end(), pa + assoc_offset);
    copy(thrust::seq, local_ta.begin(), local_ta.end(), ta + assoc_offset);
    copy(thrust::seq, local_la.begin(), local_la.end(), la + assoc_offset);
  }
}

template <typename T>
auto redistribute_pts(
  int const assoc_size,
  int const num_tetra,
  device_vector<tetra> const& mesh,
  device_vector<point_t<T>> const& pts,
  device_vector<int> const& nm,
  device_vector<int> const& fl,
  device_vector<int>& pa,
  device_vector<int>& ta,
  device_vector<int>& la) -> void
{
  device_vector<int> nm_ta{num_tetra, -1};
  device_vector<int> num_redistributions{1, 0};
  
  mark_nominated_tetra<<<bpg, tpb>>>(
    assoc_size,
    pa.data().get(),
    ta.data().get(),
    nm.data().get(),
    nm_ta.data().get());
  
  redistribute_pts_kernel<T><<<bpg, tpb>>>(
    assoc_size,
    num_tetra,
    mesh.data().get(),
    pts.data().get(),
    nm.data().get(),
    nm_ta.data().get(),
    fl.data().get(),
    pa.data().get(),
    ta.data().get(),
    la.data().get(),
    num_redistributions.data().get());
    
  cudaDeviceSynchronize();
}


#endif // REGULUS_LIB_REDISTRIBUTE_PTS_HPP_
