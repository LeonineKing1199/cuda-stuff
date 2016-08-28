#include "../include/lib/fract-locations.hpp"

/*
  Now it's time that we figure out _how_ we're going to wind up
  writing to the tetrahedron buffer.
  
  1. We only fracture nominated points
  2. The size of the fracture is the number of set bits in la[tid]
  3. If we take the exclusive_scan of the size of each fracture - 1,
     we can create accurate write indices for our fracture function
*/

auto fract_locations(
  int const* __restrict__ pa,
  int const* __restrict__ nm,
  int const* __restrict__ la,
  int const assoc_size,
  int* __restrict__ fl) -> void
{
  // Try out some fancy new CUDA stuff with "extended lambdas"
  // Create a functor such that given a tuple of two integers (pa, la),
  // we return nm[pa[tid]] * (num_set_bits(la) - 1) => (0 or 1) * fracture_size - 1
  auto fs = [=] __device__ (thrust::tuple<int const, int const> const tup) -> int
  {
    int const pa_id = thrust::get<0>(tup);
    int const la = thrust::get<1>(tup);
    
    int const fracture_size = __popc(static_cast<unsigned int>(la)) - 1;
    return nm[pa_id] * fracture_size;
  };
  
  // Not sure how this all works but Thrust seems smart enough to know
  // that a zip iterator of two pointers should be dereferenced as a
  // tuple of values as is evidenced by our use of make_transform_iterator
  // and our functor as described above
  auto const zip_begin = thrust::make_zip_iterator(
    thrust::make_tuple(
      thrust::device_ptr<int const>{pa},
      thrust::device_ptr<int const>{la}));
  
  auto const begin = thrust::make_transform_iterator(zip_begin, fs);
  
  // perform our exclusive_scan and write the result to fl
  thrust::exclusive_scan(
    thrust::device,
    begin, begin + assoc_size,
    thrust::device_ptr<int>{fl});
}