#include "globals.hpp"
#include "lib/nominate.hpp"

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

/**
  * This function is used to determine which points will
  * be used in this round of insertion.
  * nm is an array aligned to the number of points that
  * are available
  * pa, ta, la are the association tuple
*/

auto nominate(
  long long const assoc_size,
  thrust::device_vector<long long>& pa,
  thrust::device_vector<long long>& ta,
  thrust::device_vector<long long>& la,
  thrust::device_vector<long long>& nm) -> void
{
  // the first thing we want to do is sort everything
  // by ta
  auto zip_begin =
    make_zip_iterator(
      make_tuple(
        pa.begin(),
        ta.begin(),
        la.begin()));
        
  sort(
    zip_begin, zip_begin + assoc_size,
    [] __device__ (
      tuple<long long, long long, long long> const& a,
      tuple<long long, long long, long long> const& b) -> bool
    {
      long long const a_ta_id{get<1>(a)};
      long long const a_pa_id{get<0>(a)};
      
      long long const b_ta_id{get<1>(b)};
      long long const b_pa_id{get<0>(b)};
      
      return (a_ta_id == b_ta_id) ? (a_pa_id < b_pa_id) : (a_ta_id < b_ta_id);
    });
 
  
  // we then want to allocate copies of our
  // association arrays to write our stream
  // compaction to
  device_vector<long long> pa_cpy{static_cast<unsigned long long>(assoc_size), -1};
  device_vector<long long> ta_cpy{static_cast<unsigned long long>(assoc_size), -1};
    
  // remove tuple elements, using ta as the
  // unique key
  auto last_pair = unique_by_key_copy(
    ta.begin(), ta.begin() + assoc_size,
    pa.begin(),
    ta_cpy.begin(),
    pa_cpy.begin());

  // unique_by_key_copy returns a pair of iterators (keys_last, values_last)
  long long const assoc_cpy_size{static_cast<long long>(distance(ta_cpy.begin(), last_pair.first))};
  
  fill(nm.begin(), nm.end(), 0);
  device_vector<long long> nm_cpy{nm};
  
  long long* nm_data = nm.data().get();
  long long* nm_cpy_data = nm_cpy.data().get();
  
  // this is how we count the number of occurrences for a particular
  // point index
  // if the copy doesn't match up with the original count array, that
  // means that the point had some non-unique tetrahedra associated
  // with it and as such is not up for nomination
  for_each(
    pa.begin(), pa.begin() + assoc_size,
    [=] __device__ (long long const pa_id) -> void
    {
      atomicAdd(reinterpret_cast<unsigned long long*>(nm_data + pa_id), 1);
    });
    
  for_each(
    pa_cpy.begin(), pa_cpy.begin() + assoc_cpy_size,
    [=] __device__ (long long const pa_id) -> void
    {
      atomicAdd(reinterpret_cast<unsigned long long*>(nm_cpy_data + pa_id), 1);
    });
    
  // we perform a simple transformation over both ranges and
  // check for equality.
  // if the point occurred the same amount of times then all
  // of its  tetrahedra were unique and is able to be nominated
  transform(
    nm.begin(), nm.end(),
    nm_cpy.begin(),
    nm.begin(),
    [] __device__ (long long const a, long long const b) -> long long
    {
      return (a != 0) && (a - b == 0);
    });//*/
}
