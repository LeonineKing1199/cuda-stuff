#include "lib/set-internal-fract-adjacencies.hpp"

__host__ __device__
auto set_interal_fract_adjacencies(
  array<index_t, 4> const& fract_locs,
  loc_t             const  loc_code) -> array<adjacency, 4>
{
  ptrdiff_t const adjacency_map[4][3] = {
    { 3, 2, 1 },
    { 0, 2, 3 },
    { 0, 3, 1 },
    { 0, 1, 2 }
  };

  array<adjacency, 4> adj_relations;
  array<uint8_t,   4> half_space_relations;

  size_t const num_faces = 4;
  for (size_t i = 0; i < num_faces; ++i) {
    half_space_relations[i] = loc_code & (1 << i);
  }

  for (decltype(fract_locs.size()) i{}; i < fract_locs.size(); ++i)
  {
    if (half_space_relations[i]) {
      adj_relations[i] = { 
        half_space_relations[adjacency_map[i][0]] ? fract_locs[adjacency_map[i][0]] : -1,
        half_space_relations[adjacency_map[i][1]] ? fract_locs[adjacency_map[i][1]] : -1,
        half_space_relations[adjacency_map[i][2]] ? fract_locs[adjacency_map[i][2]] : -1,
        -1 };
    } else {
      adj_relations[i] = { -1, -1, -1, -1 };
    }
  }

  return adj_relations;
}