#include "lib/set-internal-fract-adjacencies.hpp"

__host__ __device__
auto set_interal_fract_adjacencies(
  array<index_t, 4> const& locations,
  loc_t const loc_code) -> array<adjacency, 4>
{
  ptrdiff_t const adjacency_map[4][3] = {
    { 3, 2, 1 },
    { 0, 2, 3 },
    { 0, 3, 1 },
    { 0, 1, 2 }
  };

  array<adjacency, 4> adj_relations;
  for (auto i = 0; i < locations.size(); ++i) {
    if (loc_code & (1 << i)) {
      ptrdiff_t const* loc_idx = adjacency_map[i];

      adj_relations[i] = {
        locations[loc_idx[0]], 
        locations[loc_idx[1]], 
        locations[loc_idx[2]], 
        -1
      };
    }
  }

  return adj_relations;
}