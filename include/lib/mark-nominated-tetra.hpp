#ifndef REGULUS_LIB_MARK_NOMINATED_TETRA_HPP_
#define REGULUS_LIB_MARK_NOMINATED_TETRA_HPP_

__global__
void mark_nominated_tetra(
  int const assoc_size,
  int const* __restrict__ pa,
  int const* __restrict__ ta,
  int const* __restrict__ nm,
  int* __restrict__ nm_ta);

#endif // REGULUS_LIB_MARK_NOMINATED_TETRA_HPP_
