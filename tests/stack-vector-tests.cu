#include <thrust/transform.h>

#include "test-suite.hpp"
#include "../include/stack-vector.hpp"
#include "../include/globals.hpp"
#include "../include/math/point.hpp"

__host__ __device__
auto stack_vector_tests_impl(void) -> void
{
  // It should be constructible
  {
    stack_vector<int, 16> v;
    assert(v.size() == 0);
    
    v.emplace_back(1337);
    assert(v.size() == 1);
  }
  
  // It should be value-type constructible and transformable
  {
    stack_vector<int, 32> vec{1};
    assert(vec.size() == 32);
 
    stack_vector<int, 32> other_vec{-1};   
    
    for (auto v : vec) {
      assert(v == 1);
    }
    
    for (int i = 0; i < other_vec.size(); ++i) {
      assert(other_vec[i] == -1);
    }
    
    thrust::transform(
      thrust::seq,
      vec.begin(), vec.end(),
      other_vec.begin(),
      [](int const v) -> int
      {
        return v * 3;
      });
      
    for (auto v : other_vec) {
      assert(v == 3);
    }
  }
  
  // It should support const references
  {
    using real = float;
    
    stack_vector<point_t<real>, 4> pts;
    point_t<real> const p{1, 2, 3};
    
    pts.push_back(p);
    
    assert(pts.size() == 1);
    assert(pts[0] == p);
    
    pts.push_back(p);
    pts.push_back(p);
    pts.push_back(p);
    
    assert(pts.size() == 4);
    
    for (auto const& pt: pts) {
      assert(pt == p);
    }
  }
}

__global__
void device_stack_vector_tests(void)
{
  stack_vector_tests_impl();
}

auto stack_vector_tests(void) -> void
{
  std::cout << "Beginning stack vector tests!" << std::endl;
  
  stack_vector_tests_impl();
  device_stack_vector_tests<<<1, 512>>>();
  cudaDeviceSynchronize();
  
  std::cout << "Tests Passed!\n" << std::endl;
}