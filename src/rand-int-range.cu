#include "math/rand-int-range.hpp"
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/detail/use_default.h>

using thrust::device_vector;
using thrust::counting_iterator;
using thrust::use_default;
using thrust::transform;
using thrust::default_random_engine;
using thrust::uniform_int_distribution;

thrust::device_vector<int> rand_int_range(
  int const min,
  int const max,
  int const num_vals,
  int const seed)
{
  device_vector<int> rand_vals{(long unsigned int ) num_vals};
  counting_iterator<int, use_default, use_default, int> it{seed};

  transform(
    it, it + num_vals,
    rand_vals.begin(),
    [=] __device__ (int const idx) -> int
    {
      default_random_engine rng;
      uniform_int_distribution<int> dist{min, max - 1};
      rng.discard(idx);
      return dist(rng);
    });
        
  return rand_vals;
}