#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>
#include <thrust/for_each.h>
#include <thrust/distance.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>

#include "regulus/algorithm/nominate.hpp"

namespace regulus
{
  // nominate points by first sorting our tuple of
  // assocation arrays (ta, pa, la) by ta then
  // write a filtered view of ta and pa
  // count the number of times a point was mentioned in
  // both pa and the filtered copy
  // if the point counts match, the point does not fracture
  // any other potential tetrahedra
  // if the point count does not match, it means the point
  // is attempting to fracture a tetrahedron that's being fractured
  // by another point
  auto nominate(
    span<std::ptrdiff_t> const pa,
    span<std::ptrdiff_t> const ta,
    span<loc_t>          const la,
    span<bool>           const nm) -> void
  {
    using thrust::get;

    auto const assoc_size = pa.size();

    // allocate buffers that we'll use to write our filtered view
    // of pa and ta to
    auto pa_copy = thrust::device_vector<std::ptrdiff_t>{assoc_size, -1};
    auto ta_copy = thrust::device_vector<std::ptrdiff_t>{assoc_size, -1};

    // allocate storage for the point counts for both views
    auto pa_id_count      = thrust::device_vector<unsigned>{nm.size(), 0};
    auto pa_id_copy_count = thrust::device_vector<unsigned>{nm.size(), 0};

    auto zip_begin = thrust::make_zip_iterator(
      thrust::make_tuple(
        ta.begin(),
        pa.begin(),
        la.begin()));

    auto pair_begin = thrust::make_zip_iterator(
      thrust::make_tuple(
        ta.begin(),
        pa.begin()));

    auto pair_copy_begin = thrust::make_zip_iterator(
      thrust::make_tuple(
        ta_copy.begin(),
        pa_copy.begin()));

    // first sort everything by ta
    thrust::sort(
      thrust::device,
      zip_begin, zip_begin + assoc_size,
      [] __device__ (auto const a, auto const b) -> bool
      {
        auto const a_ta = get<0>(a);
        auto const b_ta = get<0>(b);

        auto const a_pa = get<1>(a);
        auto const b_pa = get<1>(b);

        return (
          a_ta == b_ta
          ? (a_pa < b_pa)
          : (a_ta < b_ta));
      });

    // then filter out all pairs of (ta, pa)
    // such that ta is unique across the board
    auto pair_copy_end = thrust::unique_copy(
      thrust::device,
      pair_begin, pair_begin + assoc_size,
      pair_copy_begin,
      [] __device__ (auto const a, auto const b) -> bool
      {
        auto const a_ta = get<0>(a);
        auto const b_ta = get<0>(b);

        auto const a_pa = get<1>(a);
        auto const b_pa = get<1>(b);

        return a_ta == b_ta && a_pa != b_pa;
      });

    // next perform the point counts across our original
    // pa and our filtered version, pa_copy
    thrust::for_each(
      thrust::device,
      pa.begin(), pa.end(),
      [pa_ids = pa_id_count.data().get()] __device__ (auto const pa_id) -> void
      {
        atomicAdd(pa_ids + pa_id, 1);
      });

    thrust::for_each(
      thrust::device,
      pa_copy.begin(),
      pa_copy.begin() + thrust::distance(pair_copy_begin, pair_copy_end),
      [pa_ids = pa_id_copy_count.data().get()] __device__ (auto const pa_id) -> void
      {
        atomicAdd(pa_ids + pa_id, 1);
      });

    // finally, compare counts and write it to our array, nm
    thrust::transform(
      thrust::device,
      pa_id_count.begin(), pa_id_count.end(),
      pa_id_copy_count.begin(),
      nm.begin(),
      [] __device__ (auto const a, auto const b) -> bool
      {
        return (a > 0) && (a == b);
      });
  }
}