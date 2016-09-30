#include <thrust/device_vector.h>

#include "../include/lib/nominate.hpp"

using thrust::device_vector;

__global__
void set_15_first(
  int const assoc_size,
  int const* __restrict__ ta,
  int const* __restrict__ pa,
  int const* __restrict__ la,
  int* __restrict__ nm_ta,
  int* __restrict__ nm)
{
  for (auto tid = get_tid(); tid < assoc_size; tid += grid_stride()) {
    if (la[tid] != 15) {
      return;
    }
    
    if (atomicCAS(nm_ta + ta[tid], -1, tid) != -1) {
      atomicAnd(nm + pa[tid], 0);
    }
  }
}

__global__
void nominate(
  int const assoc_size,
  int const* __restrict__ ta,
  int const* __restrict__ pa,
  int* __restrict__ nm_ta,
  int* __restrict__ nm)
{
  for (auto tid = get_tid(); tid < assoc_size; tid += grid_stride()) { 
    if (atomicCAS(nm_ta + ta[tid], -1, tid) == -1) {
      // then this point is the first to nominate itself for
      // fracturing this tetrahedron!
            
      // we then want to nominate this point
      // but because we initialize pa to being all true, if any
      // entry was previously 0, we know it was marked false by
      // another thread so we set it back to being false
      if (atomicOr(nm + pa[tid], 1) == 0) {
        atomicAnd(nm + pa[tid], 0);
      }
    } else {
      atomicAnd(nm + pa[tid], 0);
    }
  }
}

__global__
void repair_nm_ta(
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






__global__
void proto_a(
  int const assoc_size,
  int const* __restrict__ ta,
  int const* __restrict__ pa,
  int* __restrict__ nm_tetra,
  int* __restrict__ fractured_by,
  int* __restrict__ nm_sr,
  int* __restrict__ num_pt_nominations,
  int* __restrict__ nm_pa)
{
  for (auto tid = get_tid(); tid < assoc_size; tid += grid_stride()) {
    int const ta_id{ta[tid]};
    int const pa_id{pa[tid]};
    
    if (atomicCAS(nm_tetra + ta_id, -1, tid) == -1) {
      atomicXor(fractured_by + ta_id, pa_id);
      atomicXor(nm_sr + pa_id, atomicAdd(num_pt_nominations, 1));
    } else {
      // still conflicting but the first of its kind
      if (~nm_sr[pa_id] == -1) {
        atomicAnd(nm_pa + pa_id, 0);
      } else {
        int const this_seniority{~nm_sr[pa_id]};
        int const conflict_seniority{~nm_sr[~fractured_by[ta_id]]};
        
        int const elder_idx{this_seniority < conflict_seniority ? this_seniority : conflict_seniority};
        int const junior_idx{this_seniority > conflict_seniority ? this_seniority : conflict_seniority};
        
        atomicAnd(nm_pa + junior_idx, 0);
        if (atomicOr(nm_pa + elder_idx, 1) == 0) {
          atomicAnd(nm_pa + elder_idx, 0);
        }
        
        atomicOr(fractured_by + ta_id, -1);
        atomicXor(fractured_by + ta_id, elder_idx);
      }
    }
  }
}










