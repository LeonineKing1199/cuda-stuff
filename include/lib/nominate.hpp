#ifndef REGULUS_LIB_NOMINATE_HPP_
#define REGULUS_LIB_NOMINATE_HPP_

#include "index_t.hpp"
#include <thrust/device_vector.h>

auto nominate(
  size_t const assoc_size,
  thrust::device_vector<index_t>& pa,
  thrust::device_vector<index_t>& ta,
  thrust::device_vector<index_t>& la,
  thrust::device_vector<unsigned>& nm) -> void;


#endif // REGULUS_LIB_NOMINATE_HPP_
