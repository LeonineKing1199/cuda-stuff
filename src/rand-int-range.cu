#include "../include/math/rand-int-range.hpp"

struct rand_gen
{
  int min;
  int max;
  
  __host__ __device__
  rand_gen(int const a, int const b) : min{a}, max{b} {};
  
  __host__ __device__
  auto operator()(int const idx) const -> int
  { 
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist{min, max - 1};
    rng.discard(idx);
    return dist(rng);
  }
};

auto rand_int_range(
  int const min,
  int const max,
  int const num_vals,
  int const seed) -> thrust::device_vector<int>
{
  thrust::device_vector<int> rand_vals{num_vals};
  thrust::counting_iterator<int> it{seed};
  
  thrust::transform(
    it, it + num_vals,
    rand_vals.begin(),
    rand_gen{min, max});
    
  return rand_vals;
}