#include "catch.hpp"
#include "globals.hpp"
#include "index_t.hpp"
#include "lib/nominate.hpp"
#include "math/rand-int-range.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using thrust::device_vector;
using thrust::host_vector;

TEST_CASE("The nomination function")
{
  /*size_type assoc_size = 11;
  size_type const ta_data[11] = { 0, 1, 2, 3, 2, 5, 6, 7, 8, 1, 8 };
  size_type const pa_data[11] = { 0, 0, 0, 0, 2, 2, 3, 3, 3, 4, 4 };
  
  host_vector<size_type> h_ta{ta_data, ta_data + 11};
  host_vector<size_type> h_pa{pa_data, pa_data + 11};
  
  device_vector<size_type> ta{h_ta};
  device_vector<size_type> pa{h_pa};
  device_vector<size_type> la{static_cast<usize_type>(assoc_size), -1};

  //*/
}

/*
__global__
void assert_unique(
  int const assoc_size,
  int const* __restrict__ pa,
  int const* __restrict__ nm,
  int const* __restrict__ ta,
  int* __restrict__ nm_ta)
{
  for (auto tid = get_tid(); tid < assoc_size; tid += grid_stride()) {
    if (nm[pa[tid]]) {
      assert(atomicCAS(nm_ta + ta[tid], -1, 1) == -1);
    }
  }
}

auto nomination_tests(void) -> void
{
  std::cout << "Beginning nomination tests!" << std::endl;
  
  {


    device_vector<int> nm{5, 0};

    nominate(assoc_size, pa, ta, la, nm);
    
    device_vector<int> nm_ta{9, -1};
    
    assert_unique<<<bpg, tpb>>>(
      assoc_size,
      pa.data().get(),
      nm.data().get(),
      ta.data().get(),
      nm_ta.data().get());
    
    cudaDeviceSynchronize();
    
    /*for (unsigned i = 0; i < nm.size(); ++i) {
      std::cout << nm[i] << " ";
    }
    std::cout << "\n";//
  }
  
  {
    int assoc_size{5000};
    
    int const min{0};
    int const max{2500};
    
    device_vector<int> pa{rand_int_range(min, max, assoc_size, 0)};
    device_vector<int> ta{rand_int_range(min, max, assoc_size, assoc_size)};
    device_vector<int> la{assoc_size, -1};
    
    assert(pa.size() == static_cast<unsigned>(assoc_size));
    assert(ta.size() == static_cast<unsigned>(assoc_size));
    
    int const num_pts{max};
    device_vector<int> nm{num_pts, 0};
    
    nominate(assoc_size, pa, ta, la, nm);
    
    device_vector<int> nm_ta{max, -1};
    
    assert_unique<<<bpg, tpb>>>(
      assoc_size,
      pa.data().get(),
      nm.data().get(),
      ta.data().get(),
      nm_ta.data().get());
    
    cudaDeviceSynchronize();
    
    /*for (unsigned i = 0; i < nm.size(); ++i) {
      std::cout << nm[i];
    }
    std::cout << "\n";//
  }
  
  std::cout << "Tests Passed!\n" << std::endl;
}*/