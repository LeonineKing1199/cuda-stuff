#include "gtest/gtest.h"
#include "array.hpp"
#include "globals.hpp"

#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using thrust::transform;
using thrust::device_vector;
using thrust::host_vector;

__global__
void device_tests()
{

}

TEST(ArrayType, DefaultConstructible) 
{
  int const size{16};
  array<int, size> const x{ { 0 } };
  
  EXPECT_EQ(16, x.size());
  
  for (auto const& v : x) {
    EXPECT_EQ(0, v);
  }
}

/*__host__ __device__
auto array_tests_impl(void) -> void
{
  // we should  be able to construct an array
  {
    array<float, 4> a{ 1.0f, 2.0f, 3.0f, 4.0f };
    array<float, 4> b = a;
    
    assert((a == b) && "failed `==` test");
    
    assert(a[0] == 1.0f && "failed indexed comparison at 0");
    assert(a[1] == 2.0f && "failed indexed comparison at 1");
    assert(a[2] == 3.0f && "failed indexed comparison at 2");
    assert(a[3] == 4.0f && "failed indexed comparison at 3");
    
    b[3] = 17.0f;
    
    assert((a != b) && "failed `!=` test");
  }
  
  // it should be transformable
  {
    array<float, 4> a{ 1.0f, 2.0f, 3.0f, 4.0f };
    array<float, 4> b{ 0 };

    transform(
      thrust::seq,
      a.begin(), a.end(),
      b.begin(),
      [](float const f) -> float
      {
        return f * f;
      });
    
    assert((b == array<float, 4>{ 1.0f, 4.0f, 9.0f, 16.0f }) && "Failed transformation");
  }
}

__global__
void array_test_kernel(void)
{
  array_tests_impl();
}

auto array_tests(void) -> void
{
  std::cout << "Beginning array tests!\n";
  
  array_tests_impl();

  // "we should be able to do everything on the device as well
  {
    array_test_kernel<<<1, 256>>>();
    cudaDeviceSynchronize();
  }
  
  std::cout << "Array tests passed!\n\n";
}*/


