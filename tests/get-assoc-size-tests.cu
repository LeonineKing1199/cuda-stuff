#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>

#include "test-suite.hpp"
#include "../include/lib/get-assoc-size.hpp"
#include "../include/math/rand-int-range.hpp"

auto get_assoc_size_tests(void) -> void
{
  std::cout << "Beginning get_assoc_size tests!" << std::endl;
  
  // It should sort everything out and give us the new
  // size of the association tuples
  {
    int seed = 20000;
    int const assoc_capacity = 50000;
    
    int const min = 0;
    int const max = 5000;
    
    thrust::device_vector<int> pa = rand_int_range(
      min, max, assoc_capacity, seed);
      
    seed += assoc_capacity;
      
    thrust::device_vector<int> ta = rand_int_range(
      min, max, assoc_capacity, seed);
      
    seed += assoc_capacity;
      
    thrust::device_vector<int> la = rand_int_range(
      min, max, assoc_capacity, seed);
      
    auto const f = [] __device__ (int& v) -> void
    {
      if (v > max / 2) {
        v = -1;
      }
    };
      
    thrust::for_each(pa.begin(), pa.end(), f);
    
    int const assoc_size = get_assoc_size(
      pa.data().get(),
      ta.data().get(),
      la.data().get(),
      assoc_capacity);
      
    assert(assoc_size > 0);
      
    thrust::host_vector<int> h_pa{pa};
    thrust::host_vector<int> h_ta{ta};
    
    for (int i = 0; i < assoc_size - 1; ++i) {
      assert(h_pa[i + 1] >= h_pa[i]);
      
      if (h_pa[i + 1] == h_pa[i]) {
        assert(h_ta[i + 1] >= h_ta[i]);
      }
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