#include <thrust/transform.h>
#include <thrust/execution_policy.h>

#include "test-suite.hpp"
#include "../include/array.hpp"

template <typename DerivedPolicy>
__host__ __device__
auto array_tests_impl(const thrust::detail::execution_policy_base<DerivedPolicy> exec) -> void
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

    auto distance = a.end() - a.begin();
    auto b_iter = b.begin();
    
    for (int i = 0; i < distance; ++i) {
      b_iter[i] = a[i] * a[i];      
    }
    
    assert((b == reg::array<float, 4>{ 1.0f, 4.0f, 9.0f, 16.0f }));
  }
}

__global__
void test_kernel(void)
{
  array_tests_impl(thrust::device);
}

auto array_tests(void) -> void
{
  std::cout << "Beginning array tests!" << std::endl;
  
  array_tests_impl(thrust::host);

  // we should be able to do everything on the device as well
  {
    test_kernel<<<1, 256>>>();
    cudaDeviceSynchronize();
  }
  
  std::cout << "Array tests passed!\n" << std::endl;
}