#ifndef REGULUS_ALGORITHM_NOMINATE_HPP_
#define REGULUS_ALGORITHM_NOMINATE_HPP_

#include <thrust/device_vector.h>
#include "regulus/algorithm/location.hpp"

namespace regulus
{
  auto nominate(
    size_t const assoc_size,
    thrust::device_vector<ptrdiff_t>& pa,
    thrust::device_vector<ptrdiff_t>& ta,
    thrust::device_vector<loc_t>    & la,
    thrust::device_vector<bool>     & nm) -> void;
}

#endif // REGULUS_ALGORITHM_NOMINATE_HPP_