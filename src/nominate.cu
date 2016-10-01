#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/tuple.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/unique.h>
#include <thrust/for_each.h>
#include <thrust/pair.h>
#include <thrust/distance.h>
#include <thrust/transform.h>

#include "../include/lib/nominate.hpp"

using thrust::device_vector;
using thrust::fill;
using thrust::tuple;
using thrust::get;
using thrust::make_zip_iterator;
using thrust::make_tuple;
using thrust::sort;
using thrust::unique_by_key_copy;
using thrust::for_each;
using thrust::pair;
using thrust::distance;
using thrust::transform;

auto nominate(
  int const assoc_size,
  device_vector<int>& pa,
  device_vector<int>& ta,
  device_vector<int>& la,
  device_vector<int>& nm) -> void
{
  auto zip_begin =
    make_zip_iterator(
      make_tuple(
        pa.begin(),
        ta.begin(),
        la.begin()));
        
  sort(
    zip_begin, zip_begin + assoc_size,
    [] __device__ (
      tuple<int, int, int> const& a,
      tuple<int, int, int> const& b) -> bool
    {
      int const a_ta_id{get<1>(a)};
      int const a_pa_id{get<0>(a)};
      
      int const b_ta_id{get<1>(b)};
      int const b_pa_id{get<0>(b)};
      
      return (
        a_ta_id == b_ta_id ?
          (a_pa_id < b_pa_id) :
          (a_ta_id < b_ta_id));
    });
 
  device_vector<int> pa_cpy{assoc_size, -1};
  device_vector<int> ta_cpy{assoc_size, -1};
  device_vector<int> la_cpy{assoc_size, -1}; 
    
  auto last_pair = unique_by_key_copy(
    ta.begin(), ta.end(),
    make_zip_iterator(make_tuple(pa.begin(), la.begin())),
    ta_cpy.begin(),
    make_zip_iterator(make_tuple(pa_cpy.begin(), la_cpy.begin())));

  int const assoc_cpy_size{static_cast<int>(distance(ta_cpy.begin(), last_pair.first))};
    
  sort(
    zip_begin, zip_begin + assoc_size,
    [] __device__ (
      tuple<int, int, int> const& a,
      tuple<int, int, int> const& b) -> bool
    {
      int const a_pa_id{get<0>(a)};
      int const b_pa_id{get<0>(b)};
      
      return (a_pa_id < b_pa_id);
    });
    
  auto zip_cpy_begin =
    make_zip_iterator(
      make_tuple(
        pa_cpy.begin(),
        ta_cpy.begin(),
        la_cpy.begin()));
        
  sort(
    zip_cpy_begin, zip_cpy_begin + assoc_cpy_size,
    [] __device__ (
      tuple<int, int, int> const& a,
      tuple<int, int, int> const& b) -> bool
    {
      int const a_pa_id{get<0>(a)};
      int const b_pa_id{get<0>(b)};
      
      return (a_pa_id < b_pa_id);
    });

  fill(nm.begin(), nm.end(), 0);
  device_vector<int> nm_cpy{nm};
  
  int* nm_data = nm.data().get();
  int* nm_cpy_data = nm_cpy.data().get();
  
  for_each(
    pa.begin(), pa.begin() + assoc_size,
    [=] __device__ (int const pa_id) -> void
    {
      atomicAdd(nm_data + pa_id, 1);
    });
    
  for_each(
    pa_cpy.begin(), pa_cpy.begin() + assoc_cpy_size,
    [=] __device__ (int const pa_id) -> void
    {
      atomicAdd(nm_cpy_data + pa_id, 1);
    });
    
  transform(
    nm.begin(), nm.end(),
    nm_cpy.begin(),
    nm.begin(),
    [] __device__ (int const a, int const b) -> int
    {
      return (a != 0) && (a - b == 0);
    });
}
