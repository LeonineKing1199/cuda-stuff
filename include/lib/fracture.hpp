#ifndef REGULUS_LIB_FRACTURE_HPP_
#define REGULUS_LIB_FRACTURE_HPP_

#include <thrust/device_vector.h>
using thrust::device_vector;
using tetra = int4;

auto fracture(
  int const assoc_size,
  int const num_tetra,
  device_vector<int> const& pa,
  device_vector<int> const& ta,
  device_vector<int> const& la,
  device_vector<int> const& nm,
  device_vector<int> const& fl,
  device_vector<tetra>& mesh) -> void;

#endif // REGULUS_LIB_FRACTURE_HPP_
