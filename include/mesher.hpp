#ifndef REGULUS_MESHER_HPP_
#define REGULUS_MESHER_HPP_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <array>

#include "common.hpp"
#include "domain.hpp"
#include "math.hpp"

template <typename T>
class mesher
{
public:
  using integral = int;
  using point_t = point_t<T>;
  using tetra_t = tetra_t;
  
  using host_pts_t = thrust::host_vector<point_t>;
  using device_pts_t = thrust::device_vector<point_t>;
  
  using host_tetra_t = thrust::host_vector<tetra_t>;
  using device_tetra_t = thrust::device_vector<tetra_t>;
  
  using device_int_vector_t = thrust::device_vector<integral>;
  
protected:
  host_pts_t host_pts_;
  device_pts_t device_pts_;
  
  host_tetra_t host_tetra_;
  device_tetra_t device_tetra_;
  
  device_int_vector_t ta_;
  device_int_vector_t pa_;
  device_int_vector_t la_;
  
private:
  auto gen_cartesian_mesh(int const grid_length) -> void
  {
    host_pts_ = gen_cartesian_domain<T>(grid_length);
    device_pts_ = host_pts_;
  }
  
  auto gen_root_tetrahedron(void) -> void
  {
    T const c = point_t::max_coord_value;
    
    std::array<point_t, 4> pts{
      point_t{-c, -c, -c},
      point_t{0, c, -c},
      point_t{c, -c, -c},
      point_t{0, 0, c}
    };
    
    for (auto& p : pts) {
      host_pts_.push_back(p);
    }
    
    auto const size = host_pts_.size();
    
    tetra_t t{size - 4, size - 3, size - 2, size - 1};
    
    host_tetra_.push_back(t);
    device_tetra_ = host_tetra_;
  }
  
  auto init_ta_and_pa(void) -> void
  {
    
  }
  
public:
  auto build_cartesian_mesh(int const grid_length) -> void
  {
    gen_cartesian_mesh(grid_length);
    init_ta_and_pa();
  }
  
  auto host_data(void) const -> point_t*
  {
    return host_pts_.data();
  }
  
  auto device_data(void) const -> point_t*
  {
    return device_pts_.data();
  }
  
  auto num_points(void) const -> decltype(host_pts_.size())
  {
    return host_pts_.size();
  }
};

#endif // REGULUS_MESHER_HPP_