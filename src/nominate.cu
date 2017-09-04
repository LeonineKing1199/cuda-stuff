#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>
#include <thrust/for_each.h>
#include <thrust/distance.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>

#include "regulus/algorithm/nominate.hpp"

using zip_tuple_t  = thrust::tuple<ptrdiff_t, ptrdiff_t, regulus::loc_t>;
using ta_pa_pair_t = thrust::tuple<ptrdiff_t, ptrdiff_t>;

namespace
{
  struct sort_by_ta : public thrust::binary_function<zip_tuple_t const,
                                                     zip_tuple_t const,
                                                     bool>
  {
    __host__ __device__
    auto operator()(zip_tuple_t const a, zip_tuple_t const b) -> bool
    {
      using thrust::get;

      auto const a_ta = get<0>(a);
      auto const b_ta = get<0>(b);

      auto const a_pa = get<1>(a);
      auto const b_pa = get<1>(b);

      return (
        a_ta == b_ta
        ? (a_pa < b_pa)
        : (a_ta < b_ta));
    }
  };

  struct unique_by_ta : public thrust::binary_function<ta_pa_pair_t const,
                                                       ta_pa_pair_t const,
                                                       bool>
  {
    __host__ __device__
    auto operator()(ta_pa_pair_t const a, ta_pa_pair_t const b) -> bool
    {
      using thrust::get;

      auto const a_ta = get<0>(a);
      auto const b_ta = get<0>(b);

      auto const a_pa = get<1>(a);
      auto const b_pa = get<1>(b);

      return a_ta == b_ta && a_pa != b_pa;
    }
  };

  struct count_by_pa : public thrust::unary_function<ptrdiff_t const, void>
  {
    unsigned* pa_ids;

    count_by_pa(void) = delete;
    count_by_pa(count_by_pa const& cpy) = default;
    count_by_pa(unsigned* _pa_ids)
      : pa_ids{_pa_ids}
    {}

    __device__
    auto operator()(ptrdiff_t const pt_idx) -> void
    {
      atomicAdd(pa_ids + pt_idx, 1);
    }
  };

  struct is_nominated : public thrust::binary_function<unsigned const,
                                                       unsigned const,
                                                       bool>
  {
    __host__ __device__
    auto operator()(unsigned const a, unsigned const b) -> bool
    {
      return (a > 0) && (a == b);
    }
  };
}

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
    size_t const assoc_size,
    thrust::device_vector<ptrdiff_t>& pa,
    thrust::device_vector<ptrdiff_t>& ta,
    thrust::device_vector<loc_t>    & la,
    thrust::device_vector<bool>     & nm) -> void
  {
    // allocate buffers that we'll use to write our filtered view
    // of pa and ta to
    auto pa_copy = thrust::device_vector<ptrdiff_t>{assoc_size, -1};
    auto ta_copy = thrust::device_vector<ptrdiff_t>{assoc_size, -1};

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
      zip_begin, zip_begin + assoc_size,
      sort_by_ta{});

    // then filter out all pairs of (ta, pa)
    // such that ta is unique across the board
    auto pair_copy_end = thrust::unique_copy(
      pair_begin, pair_begin + assoc_size,
      pair_copy_begin,
      unique_by_ta{});

    // next perform the point counts across our original
    // pa and our filtered version, pa_copy
    thrust::for_each(
      pa.begin(), pa.end(),
      count_by_pa{pa_id_count.data().get()});

    thrust::for_each(
      pa_copy.begin(),
      pa_copy.begin() + thrust::distance(pair_copy_begin, pair_copy_end),
      count_by_pa{pa_id_copy_count.data().get()});

    // finally, compare counts and write it to our array, nm
    thrust::transform(
      pa_id_count.begin(), pa_id_count.end(),
      pa_id_copy_count.begin(),
      nm.begin(),
      is_nominated{});
  }
}