#ifndef REGULUS_LIB_GET_ASSOC_SIZE_HPP_
#define REGULUS_LIB_GET_ASSOC_SIZE_HPP_

#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/distance.h>
#include <thrust/remove.h>
#include <thrust/fill.h>

auto get_assoc_size(
  int* __restrict__ pa,
  int* __restrict__ ta,
  int* __restrict__ la,
  int const assoc_capacity) -> int;

#endif // REGULUS_LIB_GET_ASSOC_SIZE_HPP_
