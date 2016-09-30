#ifndef REGULUS_LIB_NOMINATE_HPP_
#define REGULUS_LIB_NOMINATE_HPP_

#include <cstdio>

#include "../globals.hpp"

__global__
void set_15_first(
  int const assoc_size,
  int const* __restrict__ ta,
  int const* __restrict__ pa,
  int const* __restrict__ la,
  int* __restrict__ nm_ta,
  int* __restrict__ nm);

__global__
void nominate(
  int const assoc_size,
  int const* __restrict__ ta,
  int const* __restrict__ pa,
  int* __restrict__ nm_ta,
  int* __restrict__ nm);

__global__
void repair_nm_ta(
  int const assoc_size,
  int const* __restrict__ pa,
  int const* __restrict__ ta,
  int const* __restrict__ nm,
  int* __restrict__ nm_ta);

__global__
void proto_a(
  int const assoc_size,
  int const* __restrict__ ta,
  int const* __restrict__ pa,
  int* __restrict__ nm_tetra,
  int* __restrict__ fractured_by,
  int* __restrict__ nm_sr,
  int* __restrict__ num_pt_nominations,
  int* __restrict__ nm_pa);

#endif // REGULUS_LIB_NOMINATE_HPP_
