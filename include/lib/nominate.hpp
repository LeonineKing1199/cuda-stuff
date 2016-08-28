#ifndef REGULUS_LIB_NOMINATE_HPP_
#define REGULUS_LIB_NOMINATE_HPP_

#include "../globals.hpp"

__global__
void nominate(
  int const assoc_size,
  int const* __restrict__ ta,
  int const* __restrict__ pa,
  int* nm_ta,
  int* nm);

#endif // REGULUS_LIB_NOMINATE_HPP_
