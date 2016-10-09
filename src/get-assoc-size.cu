#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/distance.h>
#include <thrust/remove.h>
#include <thrust/fill.h>

#include "../include/lib/get-assoc-size.hpp"

// We actually want to do quite a bit with this
// function. Namely, we want to partition the
// array tuple { pa, ta, la } such that all -1
// tuples are to the right and then sort
// the valid left-handed side such that pa is
// sorted least to greatest and then for each
// same value block of pa, ta is sorted least
// to greatest.

using thrust::tuple;
using thrust::get;
using thrust::device_vector;
using thrust::make_tuple;
using thrust::make_zip_iterator;
using thrust::remove_if;
using thrust::sort;

auto get_assoc_size(
  int const assoc_capacity,
  device_vector<int> const& nm,
  device_vector<int>& pa,
  device_vector<int>& ta,
  device_vector<int>& la) -> int
{
  using int_tuple = tuple<int, int, int>;
  
  int assoc_size{0};

  auto zip_begin =
    make_zip_iterator(
      make_tuple(
        pa.begin(),   // 0
        ta.begin(),   // 1
        la.begin())); // 2

  decltype(zip_begin) zip_end = remove_if(
    zip_begin, zip_begin + assoc_capacity,
    [] __device__ (int_tuple const& a) -> bool
    {
      return get<0>(a) == -1;
    });
    
  int const* nm_data = nm.data().get();
    
  zip_end = remove_if(
    zip_begin, zip_end,
    [=] __device__ (int_tuple const& a) -> bool
    {
      int const pa_id{get<0>(a)};
      return nm_data[pa_id] == 1;
    });
    
  assoc_size = distance(zip_begin, zip_end);
  
  sort(
    zip_begin, zip_end,
    [] __device__ (
      int_tuple const& a,
      int_tuple const& b) -> bool
    {
      int const ta_a{get<1>(a)};
      int const ta_b{get<1>(b)};
      
      return ta_a == ta_b ? get<0>(a) < get<0>(b) : ta_a < ta_b;
    });//*/
      
  fill(zip_end, zip_begin + assoc_capacity, make_tuple(-1, -1, -1));
      
  return assoc_size;
}