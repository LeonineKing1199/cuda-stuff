#ifndef REGULUS_MESHER_HPP_
#define REGULUS_MESHER_HPP_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <array>

#include "math/point.hpp"
#include "math/tetra.hpp"
#include "domain.hpp"


// rewrite this in terms of raw pointers because it's more semantic
// with usage
// still keep initialization from containers though
// thrust support seems exclusive though
template <typename T>
class mesher
{
private:
  thrust::device_vector<point_t<T>> pts_;
  thrust::device_vector<tetra> tets_;
  
  
  
public:
  mesher(
    thrust::host_vector<point_t<T>> const& h_pts,
    thrust::host_vector<tetra> const& h_input_mesh)
  : pts_{h_pts}
  {
    int const new_mesh_size{(int ) h_input_mesh.size() + 8 * (int ) pts_.size()};
    tets_.reserve(new_mesh_size);
    tets_ = h_input_mesh;
  }
  
  auto triangulate(void) -> void
  {
    sort_by_peanokey<T, decltype(pts_), thrust::device_vector<peanokey>>(pts_);
    
    
  }
};

#endif // REGULUS_MESHER_HPP_