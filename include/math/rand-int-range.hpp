#ifndef REGULUS_LIB_RAND_INT_RANGE_HPP_
#define REGULUS_LIB_RAND_INT_RANGE_HPP_

#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>

struct rand_gen;

auto rand_int_range(
  int const min,
  int const max,
  int const num_vals,
  int const seed) -> thrust::device_vector<int>;

#endif //REGULUS_LIB_RAND_INT_RANGE_HPP_
