#include <thrust/transform.h>
#include <thrust/execution_policy.h>

#include "test-suite.hpp"
#include "../include/array.hpp"

__host__ __device__
auto array_tests_impl(void) -> void
{
  // we should  be able to construct an array
  {
    reg::array<float, 4> a{ 1.0f, 2.0f, 3.0f, 4.0f };
    reg::array<float, 4> b = a;
    
    assert((a == b));
    
    assert(a[0] == 1.0f);
    assert(a[1] == 2.0f);
    assert(a[2] == 3.0f);
    assert(a[3] == 4.0f);
    
    b[3] = 17.0f;
    
    assert((a != b));
  }
  
  // it should be transformable
  {
    reg::array<float, 4> a{ 1.0f, 2.0f, 3.0f, 4.0f };
    reg::array<float, 4> b{ 0 };

    thrust::transform(
      thrust::seq,
      a.begin(), a.end(),
      b.begin(),
      [](float const f) -> float
      {
        return f * f;
      });
    
    assert((b == reg::array<float, 4>{ 1.0f, 4.0f, 9.0f, 16.0f }));
  }
}

__global__
void test_kernel(void)
{
  array_tests_impl();
}

auto array_tests(void) -> void
{
  std::cout << "Beginning array tests!" << std::endl;
  
  array_tests_impl();

  // we should be able to do everything on the device as well
  {
    test_kernel<<<1, 256>>>();
    cudaDeviceSynchronize();
  }
  
  std::cout << "Array tests passed!\n" << std::endl;
}