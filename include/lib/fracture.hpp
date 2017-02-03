#ifndef REGULUS_LIB_FRACTURE_HPP_
#define REGULUS_LIB_FRACTURE_HPP_

#include "index_t.hpp"
#include "math/tetra.hpp"
#include <thrust/device_vector.h>

auto fracture(
  size_t const assoc_size,
  size_t const num_tetra,
  thrust::device_vector<index_t>  const& pa,
  thrust::device_vector<index_t>  const& ta,
  thrust::device_vector<loc_t>    const& la,
  thrust::device_vector<unsigned> const& nm,
  thrust::device_vector<index_t>  const& fl,
  thrust::device_vector<tetra>         & mesh) -> void;

#endif // REGULUS_LIB_FRACTURE_HPP_
