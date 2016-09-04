#ifndef REGULUS_LIB_FRACTURE_HPP_
#define REGULUS_LIB_FRACTURE_HPP_

#include <thrust/copy.h>

#include "../globals.hpp"
#include "../math/tetra.hpp"
#include "../array.hpp"

__global__
void fracture(
  int const assoc_size,
  int const num_tetra,
  int const* __restrict__ pa,
  int const* __restrict__ ta,
  int const* __restrict__ la,
  int const* __restrict__ nm,
  int const* __restrict__ fl,
  tetra* __restrict__ mesh);

#endif // REGULUS_LIB_FRACTURE_HPP_
