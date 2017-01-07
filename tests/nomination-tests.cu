#include "catch.hpp"
#include "globals.hpp"
#include "index_t.hpp"
#include "array.hpp"
#include "lib/nominate.hpp"
#include "math/rand-int-range.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

using thrust::device_vector;
using thrust::host_vector;
using thrust::copy;
using thrust::max_element;
using thrust::transform;
using thrust::identity;

TEST_CASE("The nomination function with a smaller data set")
{
  // allocate some basic dummy data
  size_t const assoc_size = 11;
  array<index_t, assoc_size> ta_data = { 0, 1, 2, 3, 2, 5, 6, 7, 8, 1, 8 };
  array<index_t, assoc_size> pa_data = { 0, 0, 0, 0, 2, 2, 3, 3, 3, 4, 4 };

  // create some Thrust constructs around our raw data
  host_vector<index_t> h_ta{assoc_size};
  host_vector<index_t> h_pa{assoc_size};

  copy(thrust::seq, ta_data.begin(), ta_data.end(), h_ta.begin());
  copy(thrust::seq, pa_data.begin(), pa_data.end(), h_pa.begin());

  device_vector<index_t> ta{h_ta};
  device_vector<index_t> pa{h_pa};
  device_vector<loc_t> la{assoc_size};

  // create the nomination array
  // the number of the points is simply the largest value in the pa_data
  // and because it represents an index into another array, we add 1
  size_t const num_pts = (*max_element(pa_data.begin(), pa_data.end()) + index_t{1});
  device_vector<unsigned> nm{num_pts};

  // test the main function we're after 
  nominate(assoc_size, pa, ta, la, nm);
  cudaDeviceSynchronize();

  // now we ensure that our routine is correct
  // we ensure that if a point _is_ nominated that its corresponding
  // tetrahedron is unique
  // we enforce this using a dumb counter
  size_t const num_tetra = (*max_element(ta_data.begin(), ta_data.end())).t + 1;

  host_vector<unsigned> nominated_tetra{num_tetra, 0};
  host_vector<unsigned> h_nm{nm};

  int num_nominated = 0;
  bool found_duplicate_tetra = false;

  for (size_t i = 0; i < assoc_size; ++i) {
    if (h_nm[h_pa[i]]) {
      found_duplicate_tetra = 
        found_duplicate_tetra 
        || 
        !(++nominated_tetra[h_ta[i]] == 1);
        
      ++num_nominated;
    }
  }

  REQUIRE(!found_duplicate_tetra);
  REQUIRE(num_nominated > 0);
}

TEST_CASE("The nomination function with a larger dataset")
{
  size_t const assoc_size{5000};

  int const min{0};
  int const max{2500};

  device_vector<index_t> pa{assoc_size};
  device_vector<index_t> ta{assoc_size};
  device_vector<loc_t> la{assoc_size};

  {
    auto const rand_pa = rand_int_range(min, max, assoc_size, 0);
    auto const rand_ta = rand_int_range(min, max, assoc_size, assoc_size);

    transform(
      thrust::device,
      rand_pa.begin(), rand_pa.end(),
      pa.begin(),
      identity<long long int>{});

    transform(
      thrust::device,
      rand_ta.begin(), rand_ta.end(),
      ta.begin(),
      identity<long long int>{});
  }

  size_t const num_pts = max;
  size_t const num_tetra = max;

  device_vector<unsigned> nm{num_pts, 0};

  nominate(assoc_size, pa, ta, la, nm);

  device_vector<unsigned> nm_ta{max, -1};    
  cudaDeviceSynchronize();

  host_vector<unsigned> nominated_tetra{num_tetra, 0};
  host_vector<unsigned> h_nm{nm};

  host_vector<index_t> h_pa{pa};
  host_vector<index_t> h_ta{ta};

  int num_nominated = 0;
  bool found_duplicate_tetra = false;

  for (size_t i = 0; i < assoc_size; ++i) {
    if (h_nm[h_pa[i]]) {
      found_duplicate_tetra = 
        found_duplicate_tetra 
        || 
        !(++nominated_tetra[h_ta[i]] == 1);

      ++num_nominated;
    }
  }

  REQUIRE(!found_duplicate_tetra);
  REQUIRE(num_nominated > 0);
}