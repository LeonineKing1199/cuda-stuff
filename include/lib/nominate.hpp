#ifndef REGULUS_LIB_NOMINATE_HPP_
#define REGULUS_LIB_NOMINATE_HPP_

#include <thrust/device_vector.h>
using thrust::device_vector;

auto nominate(
  int const assoc_size,
  device_vector<int>& pa,
  device_vector<int>& ta,
  device_vector<int>& la,
  device_vector<int>& nm) -> void;


#endif // REGULUS_LIB_NOMINATE_HPP_
