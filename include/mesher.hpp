#ifndef REGULUS_MESHER_HPP_
#define REGULUS_MESHER_HPP_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

#include "math/point.hpp"
#include "math/tetra.hpp"
#include "domain.hpp"
#include "globals.hpp"
#include "lib/calc-ta-and-pa.hpp"
#include "lib/nominate.hpp"
#include "lib/fract-locations.hpp"

template <typename T>
class mesher
{
public:
  using size_type = int;
  
private:
  tetra* tets_;
  
  point_t<T>* pts_;
  int* nm_;
  
  size_type* pa_;
  size_type* ta_;
  size_type* la_;
  
  size_type num_pts_;
  size_type num_tets_;
  
  size_type assoc_size_;
  size_type assoc_capacity_;
  
  auto sort_by_peanokey(void) -> void
  {
    auto const keys_begin = thrust::make_transform_iterator(
      thrust::device_ptr<point_t<T>>(pts_), peanokey_hash<T>{});
    
    thrust::device_vector<peanokey> keys{keys_begin, keys_begin + num_pts_};
    thrust::sort_by_key(keys.begin(), keys.end(), thrust::device_ptr<point_t<T>>(pts_));
  }
  
  auto init_pts(thrust::host_vector<point_t<T>> const& pts) -> void
  {
    num_pts_ = pts.size();
    cudaMalloc(&pts_, num_pts_ * sizeof(*pts_));
    thrust::copy(pts.begin(), pts.end(), thrust::device_ptr<point_t<T>>(pts_));
    
    cudaMalloc(&nm_, num_pts_ * sizeof(*nm_));
    thrust::fill(
      thrust::device_ptr<int>(nm_), thrust::device_ptr<int>(nm_ + num_pts_),
      1);
    
    sort_by_peanokey();
  }
  
  // assume num_pts_ has been constructed
  auto init_tets(tetra const t) -> void
  {
    size_type const est_num_tetra = 8 * num_pts_;
    cudaMalloc(&tets_, est_num_tetra * sizeof(*tets_));
    *thrust::device_ptr<tetra>(tets_) = t;
    num_tets_ = 1;
  }
  
  auto init_assoc(tetra const t) -> void
  {
    // eh, on average 8 associations per point seems reasonable
    // for most cases
    size_type const est_num_associations = 8 * num_pts_;
    size_type const bytes = est_num_associations * sizeof(size_type);
    
    cudaMalloc(&la_, bytes);
    cudaMalloc(&ta_, bytes);
    cudaMalloc(&pa_, bytes);
    
    calc_initial_assoc<T><<<bpg, tpb>>>(
      pts_, num_pts_,
      t,
      pa_, ta_, la_);
    
    cudaDeviceSynchronize();
    
    assoc_size_ = num_pts_;
    assoc_capacity_ = est_num_associations;
  }
  
public:
  // am I asking a constructor to do too much here?
  mesher(
    thrust::host_vector<point_t<T>> const& h_pts,
    tetra const& root_tet)
  {
    init_pts(h_pts);
    init_tets(root_tet);
    init_assoc(root_tet);
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