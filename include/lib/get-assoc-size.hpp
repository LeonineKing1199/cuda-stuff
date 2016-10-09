#ifndef REGULUS_LIB_GET_ASSOC_SIZE_HPP_
#define REGULUS_LIB_GET_ASSOC_SIZE_HPP_

#include <thrust/device_vector.h>

using thrust::device_vector;

auto get_assoc_size(
  int const assoc_capacity,
  device_vector<int> const& nm,
  device_vector<int>& pa,
  device_vector<int>& ta,
  device_vector<int>& la) -> int;

#endif // REGULUS_LIB_GET_ASSOC_SIZE_HPP_
