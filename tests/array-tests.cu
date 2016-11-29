#include "catch.hpp"
#include "array.hpp"
#include "globals.hpp"

#include <iostream>
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

TEST_CASE("array<T, N>") 
{
  SECTION("should be default-constructible")
  {
    int const size = 16;
    array<int, size> const x{ { 0 } };
    
    REQUIRE(16 == x.size());
    
    for (auto const& v : x) {
      REQUIRE(0 == v);
    }
  }
  
  SECTION("should work on the device too")
  {
    int const size = 128;
    device_vector<int> vals{ size, -1 };
    device_tests<<<bpg, tpb>>>(vals.data().get(), size);  
    cudaDeviceSynchronize();
    
    host_vector<int> h_vals{ vals };
    for (int v : h_vals) {
      REQUIRE(8128 == v);
    }       
  }
  
  SECTION("should be comparable")
  {
    int const size = 16;
    array<int, size> a{ { 0 } };
    array<int, size> b{ { 0 } };
    
    REQUIRE(a == b);
  }
  
  SECTION("should support the front and back methods")
  {
    array<int, 3> x = { 1, 2, 3 };
    
    REQUIRE(x.front() == 1);
    x.front() = 11;
    REQUIRE(x.front() == 11);
    
    REQUIRE(x.back() == 3);
    x.back() = 17;
    REQUIRE(x.back() == 17);    
  }
  
  SECTION("should support pointer-based access")
  {
    array<int, 3> const x = { 1, 2, 3 };
    typename array<int, 3>::const_pointer begin = x.data();
    REQUIRE(*begin == 1);
  }
  
  SECTION("should support const-correctness")
  {
    using size_type = typename array<int, 5>::size_type;
    array<int, 5> const a = { 1, 2, 3, 4, 5 };
    
    for (size_type i = 0; i < a.size(); ++i) {
      REQUIRE(i + 1 == a[i]);
    }  
  }
}

