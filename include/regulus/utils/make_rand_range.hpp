#ifndef REGULUS_UTILS_MAKE_RANGE_RANGE_HPP_
#define REGULUS_UTILS_MAKE_RANGE_RANGE_HPP_

#include <ctime>
#include <cstring>
#include <cstddef>
#include <type_traits>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>

namespace regulus
{
  using rng_engine_t = thrust::random::minstd_rand;

  // Write `num_vals` elements in the range of [min, max] to `output`
  template <
    typename Integral,
    typename OutputIterator>
  auto make_rand_range(
    std::size_t const num_vals,
    Integral    const min,
    Integral    const max,
    OutputIterator output) -> void
  {
    using result_type = typename rng_engine_t::result_type;

    auto seed       = result_type{0};
    auto const time = std::time(nullptr);
    std::memcpy(
      std::addressof(seed),
      std::addressof(time),
      sizeof(seed));


    auto urng = rng_engine_t{seed};
    auto dist =
      thrust::random::uniform_int_distribution<Integral>{min, max};

    for (std::size_t i = 0; i < num_vals; ++i) {
      *output = dist(urng);
      ++output;
    }
  }
}

#endif // REGULUS_UTILS_MAKE_RANGE_RANGE_HPP_