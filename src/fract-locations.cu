#include <thrust/scan.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>

#include "../include/lib/fract-locations.hpp"

/*
  Now it's time that we figure out _how_ we're going to wind up
  writing to the tetrahedron buffer.
  
  
  We want to fracture only nominated points
  We want to perform the first write of the fracture set in-place
  with the old tetrahedron
  This means that for every fracture set, we want fracure_size - 1
  extra writes to the end of the buffer.
  We can use an inclusive scan to  generate safe write indices for
  nominated points. This means that each thread needs to read from
  one value to the left in terms of fracture location array access.
  Last element is total number of new tetrahedra added to the mesh
  for this round of insertion
*/

using thrust::device_vector;
using thrust::make_zip_iterator;
using thrust::make_tuple;
using thrust::tuple;
using thrust::get;
using thrust::inclusive_scan;
using thrust::fill;

auto fract_locations(
  int const assoc_size,
  device_vector<int> const& pa,
  device_vector<int> const& nm,
  device_vector<int> const& la,
  device_vector<int>& fl) -> void
{
  /*auto const zip_begin = make_zip_iterator(make_tuple(pa.begin(), la.begin()));
  int const* nm_data = nm.data().get();
  
  auto const begin = make_transform_iterator(
    zip_begin,
    [=] __device__ (tuple<int, int> const& t) -> int
    {
      int const pa_id{get<0>(t)};
      int const la_id{get<1>(t)};
      
      int const fract_size{__popc(static_cast<unsigned>(la_id)) - 1};
      return nm_data[pa_id] * fract_size;
    });
 
  fill(fl.begin(), fl.begin() + assoc_size, -1);
  inclusive_scan(
    begin, begin + assoc_size,
    fl.begin());//*/
}