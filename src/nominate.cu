#include "globals.hpp"
#include "size_type.hpp"
#include "index_t.hpp"
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

namespace T = thrust;

/**
  * This function is used to determine which points will
  * be used in this round of insertion.
  * nm is an array aligned to the number of points that
  * are available
  * pa, ta, la are the association tuple
*/

auto nominate(
  size_t const assoc_size,
  T::device_vector<index_t>& pa,
  T::device_vector<index_t>& ta,
  T::device_vector<loc_t>& la,
  T::device_vector<unsigned>& nm) -> void
{
  // the first thing we want to do is sort everything
  // by ta
  auto zip_begin =
    T::make_zip_iterator(
      T::make_tuple(
        pa.begin(),
        ta.begin(),
        la.begin()));
        
  T::sort(
    zip_begin, zip_begin + assoc_size,
    [] __device__ (
      T::tuple<index_t, index_t, loc_t> const& a,
      T::tuple<index_t, index_t, loc_t> const& b) -> bool
    {
      index_t const a_ta_id = T::get<1>(a);
      index_t const a_pa_id = T::get<0>(a);
      
      index_t const b_ta_id = T::get<1>(b);
      index_t const b_pa_id = T::get<0>(b);
      
      return (a_ta_id == b_ta_id) 
        ? (a_pa_id < b_pa_id) 
        : (a_ta_id < b_ta_id);
    });
 
  
  // we then want to allocate copies of our
  // association arrays to write our stream
  // compaction to
  T::device_vector<index_t> pa_cpy{assoc_size};
  T::device_vector<index_t> ta_cpy{assoc_size};
    
  // remove tuple elements, using ta as the
  // unique key
  auto last_pair = T::unique_by_key_copy(
    ta.begin(), ta.begin() + assoc_size,
    pa.begin(),
    ta_cpy.begin(),
    pa_cpy.begin());

  // unique_by_key_copy returns a pair of iterators (keys_last, values_last)
  size_t const assoc_cpy_size = 
    static_cast<size_t>(T::distance(ta_cpy.begin(), last_pair.first));
  
  T::fill(nm.begin(), nm.end(), 0);
  T::device_vector<unsigned> nm_cpy{nm};
  
  unsigned* nm_data     = nm.data().get();
  unsigned* nm_cpy_data = nm_cpy.data().get();
  
  // this is how we count the number of occurrences for a particular
  // point index
  // if the copy doesn't match up with the original count array, that
  // means that the point had some non-unique tetrahedra associated
  // with it and as such is not up for nomination
  T::for_each(
    pa.begin(), pa.begin() + assoc_size,
    [=] __device__ (index_t const pa_id) -> void
    {
      atomicAdd(nm_data + pa_id, 1);
    });
    
  T::for_each(
    pa_cpy.begin(), pa_cpy.begin() + assoc_cpy_size,
    [=] __device__ (index_t const pa_id) -> void
    {
      atomicAdd(nm_cpy_data + pa_id, 1);
    });
    
  // we perform a simple transformation over both ranges and
  // check for equality.
  // if the point occurred the same amount of times then all
  // of its  tetrahedra were unique and is able to be nominated
  T::transform(
    nm.begin(), nm.end(),
    nm_cpy.begin(),
    nm.begin(),
    [] __device__ (unsigned const a, unsigned const b) -> unsigned
    {
      return static_cast<unsigned>((a != 0) && (a - b == 0));
    });
}
