#include "catch.hpp"
#include "array.hpp"
#include "index_t.hpp"
#include "math/tetra.hpp"
#include "lib/set-internal-fract-adjacencies.hpp"

TEST_CASE("The internal adjacency setting")
{
	array<index_t, 4> const fract_locs = { 1000, 1001, 1002, 1003 };

	SECTION("1 to 4 fracture")
	{
		loc_t const loc_code = 15;
		auto  const adj_relations = set_interal_fract_adjacencies(fract_locs, loc_code);

		array<adjacency, 4> expected_adj_relations = {
			adjacency{ 1003, 1002, 1001, -1 },
			adjacency{ 1000, 1002, 1003, -1 },
			adjacency{ 1000, 1003, 1001, -1 },
			adjacency{ 1000, 1001, 1002, -1 }
		};

		for (
			typename array<adjacency, 4>::size_type i{}; i < expected_adj_relations.size(); ++i)
		{
			REQUIRE(adj_relations[i] == expected_adj_relations[i]);
		}
	}
}