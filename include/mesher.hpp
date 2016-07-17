#ifndef REGULUS_MESHER_HPP_
#define REGULUS_MESHER_HPP_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "common.hpp"
#include "domain.hpp"

template <typename T>
class mesher
{
public:
  using host_pts_t = thrust::host_vector<reg::point_t<T>>;
  using device_pts_t = thrust::device_vector<reg::point_t<T>>;
  
private:
  host_pts_t host_pts;
  device_pts_t device_pts;
  
  
public:
  auto gen_cartesian_mesh(int const grid_length) -> void
  {
    host_pts = gen_cartesian_domain<T>(grid_length);
    device_pts = host_pts;
  }
  
  auto get_host_pts(void) const -> reg::point_t<T>*
  {
    return host_pts.data();
  }
  
  auto get_num_points(void) const -> decltype(host_pts.size())
  {
    return host_pts.size();
  }
};

#endif // REGULUS_MESHER_HPP_