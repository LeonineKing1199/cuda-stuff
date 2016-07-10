#ifndef REGULUS_DOMAIN_HPP_
#define REGULUS_DOMAIN_HPP_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>

#include "common.hpp"

using peanokey = long long int;

template <typename T>
auto gen_cartesian_domain(int const grid_length) -> thrust::host_vector<reg::point_t<T>>
{
  int const num_points = grid_length * grid_length * grid_length;
  thrust::host_vector<reg::point_t<T>> domain;
  domain.reserve(num_points);

  for (int x = 0; x < grid_length; ++x)
    for (int y = 0; y < grid_length; ++y)
      for (int z = 0; z < grid_length; ++z)
        domain.push_back(reg::point_t<T>{
          static_cast<T>(x), 
          static_cast<T>(y), 
          static_cast<T>(z)});

  return domain;
}

__host__ __device__
auto peano_hilbert_key(int x, int y, int z, int bits) -> peanokey;

template <typename T>
struct peanokey_hash : public thrust::unary_function<reg::point_t<T>, peanokey>
{
  using point_t = reg::point_t<T>;

  __host__ __device__
  peanokey operator()(point_t p) const
  {
    return peano_hilbert_key(p.x, p.y, p.z, 23);
  }
};

template <typename T>
auto sort_by_peanokey(thrust::device_vector<reg::point_t<T>>& domain) -> void
{
  thrust::device_vector<peanokey> keys{
    thrust::make_transform_iterator(domain.begin(), peanokey_hash<T>{}),
    thrust::make_transform_iterator(domain.end(), peanokey_hash<T>{})};
    
  thrust::sort_by_key(keys.begin(), keys.end(), domain.begin());
}

#endif // REGULUS_DOMAIN_HPP_

