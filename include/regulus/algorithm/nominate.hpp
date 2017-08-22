#ifndef REGULUS_ALGORITHM_NOMINATE_HPP_
#define REGULUS_ALGORITHM_NOMINATE_HPP_

#include <thrust/device_vector.h>
#include "regulus/algorithm/location.hpp"

namespace regulus
{
  auto nominate(
    // effective size of the associations arrays (pa, ta, la)
    size_t const size,
    size_t const num_pts,
    thrust::device_vector<ptrdiff_t> const& pa,
    thrust::device_vector<ptrdiff_t> const& ta,
    thrust::device_vector<loc_t>     const& la) -> thrust::device_vector<bool>;
}

#endif // REGULUS_ALGORITHM_NOMINATE_HPP_