#include "regulus/array.hpp"
#include "regulus/globals.hpp"

#include <catch.hpp>

#include <iostream>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>


namespace T = thrust;
namespace R = regulus;

__global__
void device_tests(int* vals, size_t const size)
{
  for (auto tid = R::get_tid(); tid < size; tid += R::grid_stride()) {
    R::array<int, 128> tmp_vals;
    
    T::sequence(T::seq, tmp_vals.begin(), tmp_vals.end());
    
    vals[tid] = T::reduce(
      T::seq,
      tmp_vals.begin(), tmp_vals.end(),
      0, 
      T::plus<int>{});
  }
}

TEST_CASE("Our array type") 
{
  SECTION("should be default-constructible")
  {
    auto const size = ptrdiff_t{16};
    R::array<int, size> const x{ { 0 } };
    
    REQUIRE(size == x.size());
    REQUIRE(
      0 == T::reduce(
        T::seq, 
        x.begin(), x.end(), 
        0, 
        T::plus<int>{}));
  }
  
  SECTION("should work on the device too")
  {
    auto const size = size_t{128};
    auto vals       = T::device_vector<int>{size, -1};
  
    device_tests<<<R::bpg, R::tpb>>>(vals.data().get(), size);  
    cudaDeviceSynchronize();
    
    auto const h_vals = T::host_vector<int>{vals};
    bool is_valid     = true;

    for (int const v : h_vals) {
      is_valid =  is_valid && (8128 == v);
    } 

    REQUIRE(is_valid);      
  }
  
  SECTION("should be comparable")
  {
    auto const size = ptrdiff_t{16};

    R::array<int, size> a{0};
    R::array<int, size> b{0};
    
    REQUIRE(a == b);
  }
  
  SECTION("should support the front and back methods")
  {
    R::array<int, 3> x = { 1, 2, 3 };
    
    REQUIRE(x.front() == 1);
    x.front() = 11;
    REQUIRE(x.front() == 11);
    
    REQUIRE(x.back() == 3);
    x.back() = 17;
    REQUIRE(x.back() == 17);    
  }
  
  SECTION("should support pointer-based access")
  {
    auto const x     = R::array<int, 3>{1, 2, 3};
    auto const begin = x.data();
    REQUIRE(*begin == 1);
  }
  
  SECTION("should support const-correctness")
  {
    using size_type = typename R::array<int, 5>::size_type;
    auto const a    = R::array<int, 5>{1, 2, 3, 4, 5};
    
    bool is_valid = true;

    for (size_type i = 0; i < a.size(); ++i) {
      is_valid = is_valid && (i + 1 == a[i]);
    }  

    REQUIRE(is_valid);
  }
}

