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

  // Sequentially write `num_vals` elements in the range of [min, max] to `output`
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

    auto const time = static_cast<result_type>(std::time(nullptr));

    auto urng = rng_engine_t{time};
    auto dist =
      thrust::random::uniform_int_distribution<Integral>{min, max};

    for (std::size_t i = 0; i < num_vals; ++i) {
      *output = dist(urng);
      ++output;
    }
  }
}

#endif // REGULUS_UTILS_MAKE_RANGE_RANGE_HPP_