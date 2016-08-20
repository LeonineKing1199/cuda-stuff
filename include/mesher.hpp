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
  point_t<T>* pts_;
  tetra* tets_;
  
  size_type* pa_;
  size_type* ta_;
  unsigned char* la_;
  bool* nm_;
  
  size_type num_pts_;
  size_type size_; // num tetrahedra
  size_type capacity_;
  
  auto alloc_ta_and_pa(void) -> void
  {
    cudaMalloc(&pa_, capacity_ * sizeof(*pa_));
    cudaMalloc(&ta_, capacity_ * sizeof(*ta_));
    cudaMalloc(&la_, capacity_ * sizeof(*la_));
    cudaMalloc(&nm_, capacity_ * sizeof(*nm_));
  }
  
public:
  // am I asking a constructor to do too much here?
  mesher(
    thrust::host_vector<point_t<T>> const& h_pts,
    tetra const& root_tet)
  : num_pts_{(int ) h_pts.size()}, size_{0}
  {
    cudaMalloc(&pts_, num_pts_ * sizeof(*pts_));
    thrust::copy(
      h_pts.begin(), h_pts.end(),
      thrust::device_ptr<point_t<T>>(pts_));
    
    // guestimate about 8 tetrahedra per point
    int const new_mesh_size{8 * num_pts_};
    
    // allocate a preemptive amount of space for our growing mesh
    cudaMalloc(&tets_, new_mesh_size * sizeof(*tets_));
    capacity_ = new_mesh_size;
    
    *thrust::device_ptr<tetra>(tets_) = root_tet;
    size_ = 1;
    
    alloc_ta_and_pa();
    
    sort_by_peanokey<T>(
      thrust::device_ptr<point_t<T>>{pts_},
      thrust::device_ptr<point_t<T>>{pts_ + num_pts_});
    
    calc_ta_and_pa<T><<<bpg, tpb>>>(
      pts_,
      root_tet,
      size_,
      la_, pa_, ta_, nm_);
    
    cudaDeviceSynchronize();
  }
  
  ~mesher(void)
  {
    cudaFree(pts_);
    cudaFree(tets_);
    cudaFree(pa_);
    cudaFree(ta_);
    cudaFree(la_);
    cudaFree(nm_);
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
    // need to select points
    // need to insert points
    // need to redistribute points
    // need to test points for delauanyhood
    // need to refine
    // need to redistribute points
  }
};

#endif // REGULUS_MESHER_HPP_