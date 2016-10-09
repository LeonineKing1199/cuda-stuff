#ifndef REGULUS_LIB_NOMINATE_HPP_
#define REGULUS_LIB_NOMINATE_HPP_

#include <thrust/device_malloc_allocator.h>

#include "../globals.hpp"

// I know this is a bad practice
// If Thrust changes their template signature,
// this code'll be broken. Good news is, if this
// code ever fails for that reason then the whole
// project can be repaired relatively simply by
// updating the template signature in every forward
// declaration
namespace thrust {
  template <typename T, typename Alloc>
  struct device_vector;
}

using thrust::device_vector;

__global__
void repair_nm_ta(
  int const assoc_size,
  int const* __restrict__ pa,
  int const* __restrict__ ta,
  int const* __restrict__ nm,
  int* __restrict__ nm_ta);


auto nominate(
  int const assoc_size,
  device_vector<int>& pa,
  device_vector<int>& ta,
  device_vector<int>& la,
  device_vector<int>& nm) -> void;


#endif // REGULUS_LIB_NOMINATE_HPP_
