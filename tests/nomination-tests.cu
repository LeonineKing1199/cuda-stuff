#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>

#include "test-suite.hpp"
#include "../include/lib/nominate.hpp"
#include "../include/globals.hpp"

int const static range_min = 0;
int const static range_max = 2500;

struct rand_gen
{
  __device__
  auto operator()(int const idx) -> int
  { 
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist{range_min, range_max};
    rng.discard(idx);
    return dist(rng);
  }
};

auto nomination_tests(void) -> void
{
  std::cout << "Beginning nomination tests!" << std::endl;
  
  // We should be able to nominate points for a given
  // association configuration
  {
    int const num_pts = 4;
    int const num_tets = 8;
    
    thrust::device_vector<int> nm{ num_pts, 1 };
    
    thrust::device_vector<int> ta{ 10 };
    thrust::device_vector<int> pa{ 10 };
    
    ta[0] = 0;
    ta[1] = 1;
    ta[2] = 2;
    ta[3] = 3;
    ta[4] = 4;
    ta[5] = 5;
    ta[6] = 0;
    ta[7] = 2;
    ta[8] = 6;
    ta[9] = 7;
    
    pa[0] = 0;
    pa[1] = 0;
    pa[2] = 0;
    pa[3] = 0;
    pa[4] = 1;
    pa[5] = 1;
    pa[6] = 2;
    pa[7] = 3;
    pa[8] = 3;
    pa[9] = 3;
    
    thrust::device_vector<int> nm_ta{num_tets, 0};
    
    nominate<<<bpg, tpb>>>(
      10,
      ta.data().get(),
      pa.data().get(),
      nm_ta.data().get(),
      nm.data().get());
      
    cudaDeviceSynchronize();
    
    assert(nm[1] == 1);
    
    if (nm[0] == 0) {
      assert(nm[2] == 1 && nm[3] == 1);
    } else {
      assert(nm[2] == 0 && nm[3] == 0);
    }
  }
  
  // Stress testing nomination routine for accuracy
  {
    int const assoc_size = 5000;
    
    thrust::device_vector<int> ta{assoc_size};
    thrust::device_vector<int> pa{assoc_size};
       
    // TODO: refactor this random number gen stuff into a helper
    thrust::transform(
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(assoc_size),
      ta.begin(),
      rand_gen{});
          
    thrust::transform(
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(assoc_size),
      pa.begin(),
      rand_gen{});
      
    thrust::device_vector<int> nm{range_max, 1};
    thrust::device_vector<int> nm_ta{range_max, 0};
     
    nominate<<<bpg, tpb>>>(
      assoc_size,
      ta.data().get(),
      pa.data().get(),
      nm_ta.data().get(),
      nm.data().get());
     
    cudaDeviceSynchronize();
    
    
  }
  
  std::cout << "Tests Passed!\n" << std::endl;
}