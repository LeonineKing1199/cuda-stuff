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
  for (
    typename array<index_t, 4>::size_type i{}; i < fract_locs.size(); ++i)
  {
    if (loc_code & (1 << i)) {
      adj_relations[i] = { fract_locs[adjacency_map[i][0]],
                           fract_locs[adjacency_map[i][1]],
                           fract_locs[adjacency_map[i][2]],
                           -1 };
    }
  }

  return adj_relations;
}