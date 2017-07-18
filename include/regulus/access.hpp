#ifndef REGULUS_ACCESS_HPP_
#define REGULUS_ACCESS_HPP_

#include "regulus/tetra_traits.hpp"

namespace regulus
{
  template <size_t N, typename Tetra>
  auto get(Tetra const& tetra) -> typename traits::coord_type<Tetra>::type
  {
    return traits::access<N, Tetra>::get(tetra);
  }

  template <size_t N, typename Tetra>
  auto set(Tetra& tetra, typename traits::coord_type<Tetra>::type value) -> void
  {
    return traits::access<N, Tetra>::set(tetra, value);
  }
}

#endif // REGULUS_ACCESS_HPP_