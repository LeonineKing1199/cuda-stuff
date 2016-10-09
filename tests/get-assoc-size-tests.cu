#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/fill.h>

#include "test-suite.hpp"
#include "../include/lib/get-assoc-size.hpp"
#include "../include/math/rand-int-range.hpp"

using thrust::device_vector;
using thrust::for_each;
using thrust::make_tuple;
using thrust::make_zip_iterator;
using thrust::get;
using thrust::host_vector;

auto get_assoc_size_tests(void) -> void
{
  std::cout << "Beginning get_assoc_size tests!" << std::endl;
  
  // It should sort everything out and give us the new
  // size of the association tuples
  {
    int seed{20000};
    int const assoc_capacity{50000};
    
    int const min{0};
    int const max{5000};
    
    device_vector<int> pa = rand_int_range(
      min, max, assoc_capacity, seed);
      
    seed += assoc_capacity;
      
    device_vector<int> ta = rand_int_range(
      min, max, assoc_capacity, seed);
      
    seed += assoc_capacity;
      
    device_vector<int> la = rand_int_range(
      min, max, assoc_capacity, seed);
      
    device_vector<int> nm{max, 0};
    int* nm_data = nm.data().get();
    for_each(
      pa.begin(), pa.end(),
      [=] __device__ (int const& v) -> void
      {
        if (v < max / 2) {
            nm_data[v] = 1;
        }
      });
      
    auto const f = [] __device__ (int& v) -> void
    {
      if (v > max / 2) {
        v = -1;
      }
    };
      
    for_each(pa.begin(), pa.end(), f);
    
    int const assoc_size{get_assoc_size(assoc_capacity, nm, pa, ta, la)};
    assert(assoc_size > 0);
      
    host_vector<int> h_pa{pa};
    host_vector<int> h_ta{ta};
    host_vector<int> h_nm{nm};
    
    for (int i = 0; i < assoc_size - 1; ++i) {
      assert(h_ta[i + 1] >= h_ta[i]);
      
      if (h_ta[i + 1] == h_ta[i]) {
        assert(h_pa[i + 1] >= h_pa[i]);
      }
      
      assert(h_nm[h_pa[i]] == 0);
    }
    
    assert(h_pa[assoc_size - 1] != -1);
    assert(h_ta[assoc_size - 1] != -1);
    
    for (int i = assoc_size; i < assoc_capacity; ++i) {
      assert(h_pa[i] == -1);
      assert(h_ta[i] == -1);
    }
  }
  
  std::cout << "Test passed!\n" << std::endl;
}