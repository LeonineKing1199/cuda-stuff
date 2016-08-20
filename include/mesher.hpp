#ifndef REGULUS_MESHER_HPP_
#define REGULUS_MESHER_HPP_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>

#include "math/point.hpp"
#include "math/tetra.hpp"
#include "domain.hpp"
#include "globals.hpp"
#include "lib/calc-ta-and-pa.hpp"

template <typename T>
class mesher
{
public:
  using size_type = int;
  
private:
  thrust::device_vector<point_t<T>> pts_;
  tetra* tets_;
  
  size_type* pa_;
  size_type* ta_;
  unsigned char* la_;
  
  size_type size_; // num tetrahedra
  size_type capacity_;
  
  auto alloc_ta_and_pa(void) -> void
  {
    cudaMalloc(&pa_, capacity_ * sizeof(*pa_));
    cudaMalloc(&ta_, capacity_ * sizeof(*ta_));
    cudaMalloc(&la_, capacity_ * sizeof(*la_));
  }
  
public:
  // am I asking a constructor to do too much here?
  mesher(
    thrust::host_vector<point_t<T>> const& h_pts,
    tetra const& root_tet)
  : pts_{h_pts}, size_{0}
  {
    // guestimate about 8 tetrahedra per point
    int const new_mesh_size{8 * (int ) pts_.size()};
    
    // allocate a preemptive amount of space for our growing mesh
    cudaMalloc(&tets_, new_mesh_size * sizeof(*tets_));
    capacity_ = new_mesh_size;
    
    *thrust::device_ptr<tetra>(tets_) = root_tet;
    size_ = 1;
    
    alloc_ta_and_pa();
    
    sort_by_peanokey<T>(pts_);
    calc_ta_and_pa<T><<<bpg, tpb>>>(
      pts_.data().get(),
      root_tet,
      size_,
      la_, pa_, ta_);
    
    cudaDeviceSynchronize();
  }
  
  ~mesher(void)
  {
    cudaFree(tets_);
    cudaFree(pa_);
    cudaFree(ta_);
    cudaFree(la_);
  }
  
  auto size(void) const -> size_type
  {
    return size_;
  }
  
  auto capacity(void) const -> size_type
  {
    return capacity_;
  }
  
  auto triangulate(void) -> void
  {
  }
};

#endif // REGULUS_MESHER_HPP_