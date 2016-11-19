#include "gtest/gtest.h"
#include "array.hpp"
#include "globals.hpp"

#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

using thrust::transform;
using thrust::device_vector;
using thrust::host_vector;
using thrust::sequence;
using thrust::reduce;
using thrust::plus;

__global__
void device_tests(int* vals, int const size)
{
  for (auto tid = get_tid(); tid < size; tid += grid_stride()) {
    array<int, 128> tmp_vals;
    
    sequence(thrust::seq, tmp_vals.begin(), tmp_vals.end());
    
    vals[tid] = reduce(
      thrust::seq, 
      tmp_vals.begin(), tmp_vals.end(), 
      0, 
      plus<int>{});
  }
}

TEST(ArrayType, DefaultConstructible) 
{
  int const size = 16;
  array<int, size> const x{ { 0 } };
  
  EXPECT_EQ(16, x.size());
  
  for (auto const& v : x) {
    EXPECT_EQ(0, v);
  }
}

TEST(ArrayType, DeviceTests)
{
  int const size = 128;
  device_vector<int> vals{ size, -1 };
  device_tests<<<bpg, tpb>>>(vals.data().get(), size);  
  cudaDeviceSynchronize();
  
  host_vector<int> h_vals{ vals };
  for (int v : h_vals) {
    EXPECT_EQ(8128, v);
  }  
}

TEST(ArrayType, Equality)
{
  int const size = 16;
  array<int, size> a{ { 0 } };
  array<int, size> b{ { 0 } };
  
  EXPECT_EQ(true, a == b);
}

TEST(ArrayType, FrontAndBack)
{
  array<int, 3> x = { 1, 2, 3 };
  
  EXPECT_EQ(x.front(), 1);
  x.front() = 11;
  EXPECT_EQ(x.front(), 11);
  
  EXPECT_EQ(x.back(), 3);
  x.back() = 17;
  EXPECT_EQ(x.back(), 17);
}

TEST(ArrayType, PointerBasedAccess)
{
  array<int, 3> const x = { 1, 2, 3 };
  
  typename array<int, 3>::const_pointer begin = x.data();
  EXPECT_EQ(*begin, 1);
}
