#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>

#include "test-suite.hpp"
#include "../include/globals.hpp"
#include "../include/lib/nominate.hpp"
#include "../include/math/rand-int-range.hpp"

int const static range_min = 0;
int const static range_max = 2500;

using thrust::device_vector;
using thrust::host_vector;
using thrust::fill;
using thrust::sort;
using thrust::sort_by_key;
using thrust::make_zip_iterator;
using thrust::make_tuple;
using thrust::tuple;
using thrust::get;
using thrust::copy_if;
using thrust::distance;
using thrust::unique_by_key_copy;
using thrust::for_each;
using thrust::pair;

__global__
void assert_unique(
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
void assert_valid_nm_ta(
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

auto nomination_tests(void) -> void
{
  std::cout << "Beginning nomination tests!" << std::endl;
  
  // We should be able to nominate points for a given
  // association configuration
  {
    int const num_pts = 4;
    int const num_tets = 8;
    
    thrust::device_vector<int> nm{ num_pts, 1 };
    
    thrust::device_vector<int> ta{ 10 };
    thrust::device_vector<int> pa{ 10 };
    
    ta[0] = 0;
    ta[1] = 1;
    ta[2] = 2;
    ta[3] = 3;
    ta[4] = 4;
    ta[5] = 5;
    ta[6] = 0;
    ta[7] = 2;
    ta[8] = 6;
    ta[9] = 7;
    
    pa[0] = 0;
    pa[1] = 0;
    pa[2] = 0;
    pa[3] = 0;
    pa[4] = 1;
    pa[5] = 1;
    pa[6] = 2;
    pa[7] = 3;
    pa[8] = 3;
    pa[9] = 3;
    
    thrust::device_vector<int> nm_ta{num_tets, -1};
    
    nominate<<<bpg, tpb>>>(
      10,
      ta.data().get(),
      pa.data().get(),
      nm_ta.data().get(),
      nm.data().get());
      
    cudaDeviceSynchronize();
    
    assert(nm[1] == 1);
    
    if (nm[0] == 0) {
      assert(nm[2] == 1 && nm[3] == 1);
    } else {
      assert(nm[2] == 0 && nm[3] == 0);
    }
  }
  
  // Stress testing nomination routine for accuracy
  {
    int const assoc_size{5000};
    
    device_vector<int> pa =
      rand_int_range(range_min, range_max, assoc_size, 0);
      
    device_vector<int> ta =
      rand_int_range(range_min, range_max, assoc_size, assoc_size);
             
    device_vector<int> nm{range_max, 1};
    device_vector<int> nm_ta{range_max, -1};
    
    for (auto t : ta) {
      assert(nm_ta[t] == -1);    
    }
    
    for (auto p : pa) {
      assert(nm[p] == 1);
    }
    
    nominate<<<bpg, tpb>>>(
      assoc_size,
      ta.data().get(),
      pa.data().get(),
      nm_ta.data().get(),
      nm.data().get());
     
    cudaDeviceSynchronize();
    
    fill(nm_ta.begin(), nm_ta.end(), -1);
    assert_unique<<<bpg, tpb>>>(
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
     
    assert_valid_nm_ta<<<bpg, tpb>>>(
      assoc_size,
      nm_ta.data().get(),
      pa.data().get(),
      ta.data().get(),
      nm.data().get());
     
    cudaDeviceSynchronize();
    
    thrust::sort_by_key(pa.begin(), pa.end(), ta.begin());
      
    /*for (int i = 0; i < assoc_size; ++i) {
      std::cout << "(" << pa[i] << ", " << ta[i] << " : " << nm[pa[i]] << std::endl;
    }//*/
  }
  
  
  
  {
    int assoc_size{11};
    int const ta_data[11] = { 0, 1, 2, 3, 2, 5, 6, 7, 8, 1, 8 };
    int const pa_data[11] = { 0, 0, 0, 0, 2, 2, 3, 3, 3, 4, 4 };
    
    host_vector<int> h_ta{ta_data, ta_data + 11};
    host_vector<int> h_pa{pa_data, pa_data + 11};
    
    device_vector<int> ta{h_ta};
    device_vector<int> pa{h_pa};

    device_vector<int> unq_ta{assoc_size, -1};
    device_vector<int> unq_pa{assoc_size, -1};

    int const num_pts{5};    
    device_vector<int> num_pt_assocs{num_pts, 0};
    device_vector<int> num_unq_pt_assocs{num_pts, 0};

    auto const zip_begin =
      make_zip_iterator(
        make_tuple(
          pa.begin(),
          ta.begin()));
    
    sort(
      zip_begin, zip_begin + assoc_size,
      [] __device__ (
        tuple<int, int> const& a,
        tuple<int, int> const& b) -> bool
      {
        return get<1>(a) < get<1>(b);
      });
    
    auto new_last = unique_by_key_copy(
      ta.begin(), ta.end(),
      pa.begin(),
      unq_ta.begin(),
      unq_pa.begin());
    
    int const unq_size{distance(unq_pa.begin(), get<1>(new_last))};
      
    for (int i = 0; i < 11; ++i) {
      std:: cout << "(pa, ta) : " << unq_pa[i] << ", " << unq_ta[i] << "\n";
    }
    std::cout << std::endl;
    
    int* pt_cnt{num_pt_assocs.data().get()};
    int* unq_pt_cnt{num_unq_pt_assocs.data().get()};
    
    for_each(
      pa.begin(), pa.end(),
      [=] __device__ (int const pa_id) -> void
      {
        atomicAdd(pt_cnt + pa_id, 1);
      });
      
    for_each(
      unq_pa.begin(), unq_pa.begin() + unq_size,
      [=] __device__ (int const pa_id) -> void
      {
        atomicAdd(unq_pt_cnt + pa_id, 1);
      });
      
    std::cout << "Before:\n";
    for (unsigned i = 0; i < num_pt_assocs.size(); ++i) {
      std::cout << "(pa id, cnt ) => " << i << ", " << num_pt_assocs[i] << "\n";
    }
    std::cout << "\n";
    
    std::cout << "After:\n";
    for (unsigned i = 0; i < num_unq_pt_assocs.size(); ++i) {
      std::cout << "(pa id, cnt ) => " << i << ", " << num_unq_pt_assocs[i] << "\n";
    }
    std::cout << "\n";
  }
  
  std::cout << "Tests Passed!\n" << std::endl;
}