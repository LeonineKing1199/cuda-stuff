#ifndef REGULUS_LIB_SET_INTERNAL_FRACT_ADJACENCIES_HPP_
#define REGULUS_LIB_SET_INTERNAL_FRACT_ADJACENCIES_HPP_

#include "array.hpp"
#include "index_t.hpp"
#include "math/tetra.hpp"

__host__ __device__
auto set_interal_fract_adjacencies(
  array<index_t, 4> const& fract_locs,
  loc_t             const  loc_code) -> array<adjacency, 4>;

#endif // REGULUS_LIB_SET_INTERNAL_FRACT_ADJACENCIES_HPP_