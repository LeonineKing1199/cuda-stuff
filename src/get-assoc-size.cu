#include "../include/lib/get-assoc-size.hpp"

// We actually want to do quite a bit with this
// function. Namely, we want to partition the
// array tuple { pa, ta, la } such that all -1
// tuples are to the right and then sort
// the valid left-handed side such that pa is
// sorted least to greatest and then for each
// same value block of pa, ta is sorted least
// to greatest.

auto get_assoc_size(
  int* pa,
  int* ta,
  int* la,
  int const assoc_capacity) -> int
{
  using int_tuple = thrust::tuple<int, int, int>;
  
  int assoc_size = 0;

  auto begin = thrust::make_zip_iterator(thrust::make_tuple(pa, ta, la));

  decltype(begin) new_last = thrust::remove_if(
    thrust::device,
    begin, begin + assoc_capacity,
    [] __device__ (int_tuple const& a) -> bool
    {
      return thrust::get<0>(a) == -1;
    });
    
  assoc_size = thrust::distance(begin, new_last);
  
  thrust::sort(
    thrust::device,
    begin, begin + assoc_size,
    [] __device__ (
      int_tuple const& a,
      int_tuple const& b) -> bool
    {
      int const a_pa = thrust::get<0>(a);
      int const b_pa = thrust::get<0>(b);
      
      if (a_pa < b_pa) {
        return true;
      }
      
      if (a_pa > b_pa) {
        return false;
      }
      
      if (a_pa == b_pa) {
        int const a_ta = thrust::get<1>(a);
        int const b_ta = thrust::get<1>(b);
        
        return a_ta < b_ta;
      }
      
      return true;
    });
      
  return assoc_size;
}