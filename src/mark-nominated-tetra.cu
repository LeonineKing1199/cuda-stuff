#include "globals.hpp"
#include "index_t.hpp"
#include "lib/mark-nominated-tetra.hpp"


__global__
void mark_nominated_tetra(
  size_t const assoc_size,
  index_t  const* __restrict__ pa,
  index_t  const* __restrict__ ta,
  unsigned const* __restrict__ nm,
  index_t       * __restrict__ nm_ta)
{
  for (auto tid = get_tid(); tid < assoc_size; tid += grid_stride()) {
    index_t const pa_id = pa[tid];
    index_t const ta_id = ta[tid];

    nm_ta[ta_id] = (nm[pa_id] == 1U) 
      ? index_t{static_cast<typename index_t::value_type>(tid)} 
      : index_t{-1};
  }
}