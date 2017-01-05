#ifndef REGULUS_LIB_FRACT_LOCATIONS_HPP_
#define REGULUS_LIB_FRACT_LOCATIONS_HPP_

#include "index_t.hpp"
#include <thrust/device_vector.h>

auto fract_locations(
  int const assoc_size,
  thrust::device_vector<index_t> const& pa,
  thrust::device_vector<unsigned> const& nm,
  thrust::device_vector<loc_t> const& la,
  thrust::device_vector<index_t>& fl) -> void;

#endif // REGULUS_LIB_FRACT_LOCATIONS_HPP_
