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
#include <cassert>

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

using thrust::host_vector;
using thrust::device_vector;
using thrust::make_zip_iterator;
using thrust::make_transform_iterator;
using thrust::make_tuple;
using thrust::copy;
using thrust::sort_by_key;
using thrust::device_ptr;
using thrust::tuple;
using thrust::get;
using thrust::remove_if;
using thrust::distance;
using thrust::fill;


// Okay, so bugs do exist and maybe asserts do need to be integrated
// into the project at the moment T_T
__global__
void assert_unique_mesher(
  int const assoc_size,
  int const* __restrict__ pa,
  int const* __restrict__ nm,
  int const* __restrict__ ta,
  int* __restrict__ nm_ta)
{
  for (auto tid = get_tid(); tid < assoc_size; tid += grid_stride()) {
    if (nm[pa[tid]]) {
      assert(atomicCAS(nm_ta + ta[tid], -1, 1) == -1);
    }
  }
}

__global__
void assert_valid_nm_ta_mesher(
  int const assoc_size,
  int const* __restrict__ nm_ta,
  int const* __restrict__ pa,
  int const* __restrict__ ta,
  int const* __restrict__ nm)
{
  for (auto tid = get_tid(); tid < assoc_size; tid += grid_stride()) {
    if (nm[pa[tid]] == 1) {
      assert(nm_ta[ta[tid]] == tid);    
    }
  }
}

template <typename T>
__global__
void assert_positivity(
  tetra const* mesh,
  point_t<T> const* pts,
  int const num_tets)
{
  for (auto tid = get_tid(); tid < num_tets; tid += grid_stride()) {
    tetra const t = mesh[tid];
    
    auto const a = pts[t.x];
    auto const b = pts[t.y];
    auto const c = pts[t.z];
    auto const d = pts[t.w];
    
    assert(orient<T>(a, b, c, d) != orientation::negative);
    assert(orient<T>(a, b, c, d) != orientation::zero);
    assert(orient<T>(a, b, c, d) == orientation::positive);
  }
}

template <typename T>
__global__
void assert_mesher_associations(
  int const* __restrict__ pa,
  int const* __restrict__ ta,
  int const* __restrict__ la,
  tetra const* __restrict__ mesh,
  point_t<T> const* __restrict__ pts,
  int const assoc_size)
{
  for (auto tid = get_tid(); tid < assoc_size; tid += grid_stride()) {
    int const pa_id = pa[tid];
    int const ta_id = ta[tid];
    int const la_id = la[tid];
    
    assert(pa_id != -1);
    assert(ta_id != -1);
    
    tetra const t = mesh[ta_id];
    auto const a = pts[t.x];
    auto const b = pts[t.y];
    auto const c = pts[t.z];
    auto const d = pts[t.w];
    
    assert(orient<T>(a, b, c, d) == orientation::positive);
    
    auto const p = pts[pa_id];
    
    assert(loc<T>(a, b, c, d, p) != -1);
    assert(loc<T>(a, b, c, d, p) != 0);
    assert(loc<T>(a, b, c, d, p) == la_id);
  }
}







// Main interface for the entire project
template <typename T>
class mesher
{
private:
  // We internally represent a mesh as a collection
  // of points.
  // Each tetra structure contains 4 indices, each one
  // corresponding to a point in the pts_ array
  device_vector<point_t<T>> pts_;
  tetra* tetra_;
  
  int num_pts_;
  int num_tetra_;
  
  // sort the internal point set by its peanokey
  auto sort_by_peanokey(void) -> void
  {
    auto const keys_begin = make_transform_iterator(
      pts_.begin(), peanokey_hash<T>{});
    
    device_vector<peanokey> keys{keys_begin, keys_begin + num_pts_};
    
    sort_by_key(keys.begin(), keys.end(), pts_.begin());
  }
    
  // assume num_pts_ has been constructed
  auto init_tets(tetra const t) -> void
  {
    int const est_num_tetra = 8 * num_pts_;
    cudaMalloc(&tetra_, est_num_tetra * sizeof(*tetra_));
    *thrust::device_ptr<tetra>(tetra_) = t;
    num_tetra_ = 1;
  }
    
public:
  // am I asking a constructor to do too much here?
  // is Thrust-only support too exclusive?
  mesher(
    host_vector<point_t<T>> const& h_pts,
    tetra const& root_tet)
  : pts_{h_pts}
  {
    // the last 4 points are the root tetrahedron
    num_pts_ = pts_.size() - 4;
    sort_by_peanokey();
    
    init_tets(root_tet);
  }
  
  ~mesher(void)
  {
    cudaFree(tetra_);
  }
    
  auto triangulate(void) -> void
  {
    // allocate storage for the association buffers...
    int const assoc_capacity = 8 * num_pts_;
    device_vector<int> pa{assoc_capacity, -1};
    device_vector<int> ta{assoc_capacity, -1};
    device_vector<int> la{assoc_capacity, -1};
    
    // then build the initial associations
    {
      tetra const t = *device_ptr<tetra>(tetra_);
    
      calc_initial_assoc<T><<<bpg, tpb>>>(
        pts_.data().get(),
        num_pts_, t,
        pa.data().get(),
        ta.data().get(),
        la.data().get());
      
      cudaDeviceSynchronize();  
    }
        
    int assoc_size{num_pts_};
    
    // need to select points
    // need to insert points
    // need to redistribute points
    // need to test points for delauanyhood
    // need to refine
    // need to redistribute points
    while (assoc_size != 0) {
      std::cout << "Allocating temporary buffers..." << std::endl;
      
      device_vector<int> nm{num_pts_, 1};
      device_vector<int> nm_ta{num_tetra_, -1};
      device_vector<int> fl{assoc_capacity, -1};
      device_vector<int> num_redistributions{1, 0};
      
      std::cout << "Nominating and repairing points..." << std::endl;
      
      nominate<<<bpg, tpb>>>(
        assoc_size,
        ta.data().get(),
        pa.data().get(),
        nm_ta.data().get(),
        nm.data().get());
      
      cudaDeviceSynchronize();
      
      thrust::fill(nm_ta.begin(), nm_ta.end(), -1);
      assert_unique_mesher<<<bpg, tpb>>>(
        assoc_size,
        pa.data().get(),
        nm.data().get(),
        ta.data().get(),
        nm_ta.data().get());
      
      repair_nm_ta<<<bpg, tpb>>>(
        assoc_size,
        pa.data().get(),
        ta.data().get(),
        nm.data().get(),
        nm_ta.data().get());
            
      assert_valid_nm_ta_mesher<<<bpg, tpb>>>(
        assoc_size,
        nm_ta.data().get(),
        pa.data().get(),
        ta.data().get(),
        nm.data().get());
      
      cudaDeviceSynchronize();
      
      std::cout << "Calculating fracture locations" << std::endl;
      
      fract_locations(
        pa.data().get(),
        nm.data().get(),
        la.data().get(),
        assoc_size,
        fl.data().get());
          
      int const last_idx = assoc_size - 1;
      int const num_new_tets =
        fl[last_idx] +
        (nm[pa[last_idx]] *
          (std::bitset<32>{la[last_idx]}.count() - 1));
          
      std::cout << "Fracturing and redistributing points" << std::endl;
          
      fracture<<<bpg, tpb>>>(
        assoc_size, num_tetra_,
        pa.data().get(),
        ta.data().get(),
        la.data().get(),
        nm.data().get(),
        fl.data().get(),
        tetra_);
      
      redistribute_pts<T><<<bpg, tpb>>>(
        assoc_size, num_tetra_,
        tetra_,
        pts_.data().get(),
        nm.data().get(),
        nm_ta.data().get(),
        fl.data().get(),
        pa.data().get(),
        ta.data().get(),
        la.data().get(),
        num_redistributions.data().get());
      
      cudaDeviceSynchronize();

      std::cout << "Calculating new association sizes" << std::endl;

      std::cout << "Stage 1 filtering..." << std::endl;

      assoc_size = get_assoc_size(
        pa.data().get(),
        ta.data().get(),
        la.data().get(),
        assoc_capacity);
      
      std::cout << "Stage 2 filtering..." << std::endl;
      
      auto zip_begin = make_zip_iterator(
        make_tuple(
          pa.begin(),
          ta.begin(),
          la.begin()));
      
      auto const* raw_nm = nm.data().get();
      
      assoc_size = distance(zip_begin, remove_if(
        thrust::device,
        zip_begin, zip_begin + assoc_size,
        [=] __device__ (tuple<int, int, int> const& t) -> bool
        {
          return raw_nm[get<0>(t)] == 1;
        }));
      
      std::cout << "Filling..." << std::endl;
      
      fill(
        thrust::device,
        zip_begin + assoc_size, zip_begin + assoc_capacity,
        make_tuple(-1, -1, -1));
      
      num_tetra_ += num_new_tets;
      
      std::cout << "number of tetrahedra: " << num_tetra_ << std::endl;
      std::cout << "size of associations: " << assoc_size << "\n" << std::endl;
      
      assert_mesher_associations<T><<<bpg, tpb>>>(
        pa.data().get(),
        ta.data().get(),
        la.data().get(),
        tetra_,
        pts_.data().get(),
        assoc_size);
      
      assert_positivity<T><<<bpg, tpb>>>(tetra_, pts_.data().get(), num_tetra_);
      
      cudaDeviceSynchronize();
    }
  }
};

#endif // REGULUS_MESHER_HPP_