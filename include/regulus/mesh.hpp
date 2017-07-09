#ifndef REGULUS_MESH_HPP_
#define REGULUS_MESH_HPP_

#include <thrust/device_vector.h>

#include "regulus/point_traits.hpp"


template <typename Point>
struct mesh
{
private:
  thrust::device_vector<Point> d_pts_;

public:
  template <typename InputIterator>
  mesh(InputIterator begin, InputIterator end)
    : d_pts_{begin, end}
  {}
};

#endif // REGULUS_MESH_HPP_