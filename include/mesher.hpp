#ifndef REGULUS_MESHER_HPP_
#define REGULUS_MESHER_HPP_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/extrema.h>
#include <bitset>

#include "math/point.hpp"
#include "math/tetra.hpp"
#include "domain.hpp"
#include "globals.hpp"
#include "lib/calc-ta-and-pa.hpp"
#include "lib/nominate.hpp"
#include "lib/fract-locations.hpp"
#include "lib/fracture.hpp"
#include "lib/redistribute-pts.hpp"
#include "lib/get-assoc-size.hpp"

template <typename T>
class mesher
{
private:
  // We internally represent a mesh as a collection
  // of points.
  // Each tetra structure contains 4 indices, each one
  // corresponding to a point in the pts_ array
  point_t<T>* pts_;
  tetra* tets_;
  
  int num_pts_;
  int num_tets_;
  
  // pa = point association
  // ta = tetrahedral association
  // la = location association
  //
  // pa[i] means point p at pts_[i] is
  // in the tetrahedron tets_[ta[i]] with
  // a location association of la[i]
  int* pa_;
  int* ta_;
  int* la_;
    
  int assoc_size_;
  int assoc_capacity_;
  
  // sort the internal point set by its peanokey
  auto sort_by_peanokey(void) -> void
  {
    auto const keys_begin = thrust::make_transform_iterator(
      thrust::device_ptr<point_t<T>>(pts_), peanokey_hash<T>{});
    
    thrust::device_vector<peanokey> keys{keys_begin, keys_begin + num_pts_};
    thrust::sort_by_key(
      thrust::device,
      keys.begin(), keys.end(),
      pts_);
  }
  
  // we construct our point set with a user-supplied one and sort it
  // automatically by its peanokey
  auto init_pts(thrust::host_vector<point_t<T>> const& pts) -> void
  {
    num_pts_ = pts.size();
    cudaMalloc(&pts_, num_pts_ * sizeof(*pts_));
    thrust::copy(
      thrust::device,
      pts.begin(), pts.end(),
      pts_);
        
    num_pts_ -= 4;
        
    sort_by_peanokey();
  }
  
  // assume num_pts_ has been constructed
  auto init_tets(tetra const t) -> void
  {
    int const est_num_tetra = 8 * num_pts_;
    cudaMalloc(&tets_, est_num_tetra * sizeof(*tets_));
    *thrust::device_ptr<tetra>(tets_) = t;
    num_tets_ = 1;
  }
  
  auto init_assoc(tetra const t) -> void
  {
    // eh, on average 8 associations per point seems reasonable
    // for most cases
    int const est_num_associations = 8 * num_pts_;
    int const bytes = est_num_associations * sizeof(int);
    
    cudaMalloc(&la_, bytes);
    cudaMalloc(&ta_, bytes);
    cudaMalloc(&pa_, bytes);
    
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(pa_, ta_, la_));
    
    thrust::fill(
      thrust::device,
      begin, begin + est_num_associations,
      thrust::make_tuple(-1, -1, -1));
    
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
  // is Thrust-only support too exclusive?
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
  }
    
  auto triangulate(void) -> void
  {
    // need to select points
    // need to insert points
    // need to redistribute points
    // need to test points for delauanyhood
    // need to refine
    // need to redistribute points
    while (assoc_size_ != 0) {
      std::cout << "Allocating temporary buffers..." << std::endl;
      
      thrust::device_vector<int> nm{num_pts_, 1};
      thrust::device_vector<int> nm_ta{num_tets_, -1};
      thrust::device_vector<int> fl{assoc_size_, -1};
      
      int* num_redistributions;
      cudaMalloc(&num_redistributions, sizeof(*num_redistributions));
      cudaMemset(num_redistributions, 0, sizeof(*num_redistributions));
      cudaDeviceSynchronize();
      
      std::cout << "Nominating and repairing points..." << std::endl;
      
      nominate<<<bpg, tpb>>>(
        assoc_size_, ta_, pa_,
        nm_ta.data().get(),
        nm.data().get());
      
      repair_nm_ta<<<bpg, tpb>>>(
        assoc_size_, pa_, ta_,
        nm.data().get(),
        nm_ta.data().get());
      
      cudaDeviceSynchronize();
      
      std::cout << "Calculating fracture locations" << std::endl;
      
      fract_locations(pa_, nm.data().get(), la_, assoc_size_, fl.data().get());
          
      std::cout << "Fracturing and redistributing points" << std::endl;
          
      fracture<<<bpg, tpb>>>(
        assoc_size_, num_tets_,
        pa_, ta_, la_,
        nm.data().get(),
        fl.data().get(),
        tets_);
      
      redistribute_pts<T><<<bpg, tpb>>>(
        assoc_size_, num_tets_,
        tets_,
        pts_,
        nm.data().get(),
        nm_ta.data().get(),
        fl.data().get(),
        pa_, ta_, la_,
        num_redistributions);
      
      cudaDeviceSynchronize();

      std::cout << "Calculating new association sizes" << std::endl;

      assoc_size_ = get_assoc_size(
        pa_, ta_, la_,
        assoc_capacity_);
      
      auto zip_begin = thrust::make_zip_iterator(
        thrust::make_tuple(pa_, ta_, la_));
      
      auto const* raw_nm = nm.data().get();
      
      assoc_size_ = thrust::distance(zip_begin, thrust::remove_if(
        thrust::device,
        zip_begin, zip_begin + assoc_size_,
        [=] __device__ (thrust::tuple<int, int, int> const& t) -> bool
        {
          return raw_nm[thrust::get<0>(t)] == 1;
        }));
      
      thrust::fill(
        thrust::device,
        zip_begin + assoc_size_, zip_begin + assoc_capacity_,
        thrust::make_tuple(-1, -1, -1));
      
      thrust::device_ptr<int> new_max{thrust::max_element(thrust::device, ta_, ta_ + assoc_size_)};
      num_tets_ = *new_max + 1;
      
      std::cout << "number of tetrahedra: " << num_tets_ << "\n" << std::endl;
      
      cudaFree(num_redistributions);
    }
  }
};

#endif // REGULUS_MESHER_HPP_