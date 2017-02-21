#ifndef REGULUS_LIB_MARK_NOMINATED_TETRA_HPP_
#define REGULUS_LIB_MARK_NOMINATED_TETRA_HPP_

__global__
void mark_nominated_tetra(
  size_t const assoc_size,
  index_t  const* __restrict__ pa,
  index_t  const* __restrict__ ta,
  unsigned const* __restrict__ nm,
  index_t       * __restrict__ nm_ta);

#endif // REGULUS_LIB_MARK_NOMINATED_TETRA_HPP_
