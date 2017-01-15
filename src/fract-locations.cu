#include "../include/lib/fract-locations.hpp"
#include <thrust/scan.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>

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
using thrust::unary_function;

struct fract_size_functor : public unary_function<tuple<index_t, loc_t> const&, index_t>
{ 
  unsigned const *nm;
  
  fract_size_functor(void) = delete;
  fract_size_functor(unsigned const *nm_) : nm{nm_} {}
  
  __device__
  auto operator()(tuple<index_t, loc_t> const &t) -> index_t
  {
    index_t const pa_id = get<0>(t);
    loc_t   const la_id = get<1>(t);
    
    index_t const fract_size{__popc(la_id) - 1};
    return {static_cast<typename index_t::value_type>(nm[pa_id] * fract_size)};
  }
};

auto fract_locations(
  size_t const assoc_size,
  device_vector<index_t>  const& pa,
  device_vector<unsigned> const& nm,
  device_vector<loc_t>    const& la,
  device_vector<index_t>& fl) -> void
{
  auto const zip_begin = make_zip_iterator(make_tuple(pa.begin(), la.begin()));
  unsigned const* nm_data = nm.data().get();
  
  auto const begin = make_transform_iterator(
    zip_begin, fract_size_functor{nm_data});
 
  fill(fl.begin(), fl.end(), index_t{-1});
  inclusive_scan(
    begin, begin + assoc_size,
    fl.begin());
}
