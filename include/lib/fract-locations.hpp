#ifndef REGULUS_LIB_FRACT_LOCATIONS_HPP_
#define REGULUS_LIB_FRACT_LOCATIONS_HPP_

#include <thrust/device_vector.h>
using thrust::device_vector;

auto fract_locations(
  int const assoc_size,
  device_vector<int> const& pa,
  device_vector<int> const& nm,
  device_vector<int> const& la,
  device_vector<int>& fl) -> void;

#endif // REGULUS_LIB_FRACT_LOCATIONS_HPP_
