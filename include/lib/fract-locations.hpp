#ifndef REGULUS_LIB_FRACT_LOCATIONS_HPP_
#define REGULUS_LIB_FRACT_LOCATIONS_HPP_

#include <thrust/scan.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

auto fract_locations(
  int const* __restrict__ pa,
  int const* __restrict__ nm,
  int const* __restrict__ la,
  int const assoc_size,
  int* __restrict__ fl) -> void;

#endif // REGULUS_LIB_FRACT_LOCATIONS_HPP_
