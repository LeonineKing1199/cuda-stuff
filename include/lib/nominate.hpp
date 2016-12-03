#ifndef REGULUS_LIB_NOMINATE_HPP_
#define REGULUS_LIB_NOMINATE_HPP_

#include <thrust/device_vector.h>

auto nominate(
  long long const assoc_size,
  thrust::device_vector<long long>& pa,
  thrust::device_vector<long long>& ta,
  thrust::device_vector<long long>& la,
  thrust::device_vector<long long>& nm) -> void;


#endif // REGULUS_LIB_NOMINATE_HPP_
