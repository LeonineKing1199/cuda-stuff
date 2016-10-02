#ifndef REGULUS_LIB_FRACTURE_HPP_
#define REGULUS_LIB_FRACTURE_HPP_

using tetra = int4;

namespace thrust {
  template <typename T, typename Alloc>
  class device_vector;
}

using thrust::device_vector;

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
