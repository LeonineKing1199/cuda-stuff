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
    loc_t const loc_code      = 15;
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

  // Face tests

  SECTION("1-to-3 flip (face 0)")
  {
    loc_t const loc_code      = 14;
    auto  const adj_relations = set_interal_fract_adjacencies(fract_locs, loc_code);

    array<adjacency, 4> expected_adj_relations = {
      adjacency{ -1,   -1,   -1, -1 },
      adjacency{ -1, 1002, 1003, -1 },
      adjacency{ -1, 1003, 1001, -1 },
      adjacency{ -1, 1001, 1002, -1 }
    };

    for (
      typename array<adjacency, 4>::size_type i{}; i < expected_adj_relations.size(); ++i)
    {
      REQUIRE(adj_relations[i] == expected_adj_relations[i]);
    }
  }

  SECTION("1-to-3 flip (face 1)")
  {
    loc_t const loc_code      = 13;
    auto  const adj_relations = set_interal_fract_adjacencies(fract_locs, loc_code);

    array<adjacency, 4> expected_adj_relations = {
      adjacency{ 1003, 1002,   -1, -1 },
      adjacency{ -1,     -1,   -1, -1 },
      adjacency{ 1000, 1003,   -1, -1 },
      adjacency{ 1000,   -1, 1002, -1 }
    };

    for (
      typename array<adjacency, 4>::size_type i{}; i < expected_adj_relations.size(); ++i)
    {
      REQUIRE(adj_relations[i] == expected_adj_relations[i]);
    }
  }  

  SECTION("1-to-3 flip (face 2)")
  {
    loc_t const loc_code      = 11;
    auto  const adj_relations = set_interal_fract_adjacencies(fract_locs, loc_code);

    array<adjacency, 4> expected_adj_relations = {
      adjacency{ 1003,   -1, 1001, -1 },
      adjacency{ 1000,   -1, 1003, -1 },
      adjacency{   -1,   -1,   -1, -1 },
      adjacency{ 1000, 1001,   -1, -1 }
    };

    for (
      typename array<adjacency, 4>::size_type i{}; i < expected_adj_relations.size(); ++i)
    {
      REQUIRE(adj_relations[i] == expected_adj_relations[i]);
    }
  }  

  SECTION("1-to-3 flip (face 3)")
  {
    loc_t const loc_code      = 7;
    auto  const adj_relations = set_interal_fract_adjacencies(fract_locs, loc_code);

    array<adjacency, 4> expected_adj_relations = {
      adjacency{   -1, 1002, 1001, -1 },
      adjacency{ 1000, 1002,   -1, -1 },
      adjacency{ 1000,   -1, 1001, -1 },
      adjacency{   -1,   -1,   -1, -1 }
    };

    for (
      typename array<adjacency, 4>::size_type i{}; i < expected_adj_relations.size(); ++i)
    {
      REQUIRE(adj_relations[i] == expected_adj_relations[i]);
    }
  }  

  // Edge tests

  SECTION("1-to-2 flip (edge 01)")
  {
    loc_t const loc_code      = 12;
    auto  const adj_relations = set_interal_fract_adjacencies(fract_locs, loc_code);
 
    array<adjacency, 4> expected_adj_relations = {
      adjacency{ -1,   -1,   -1, -1 },
      adjacency{ -1,   -1,   -1, -1 },
      adjacency{ -1, 1003,   -1, -1 },
      adjacency{ -1,   -1, 1002, -1 }
    };

    for (
      typename array<adjacency, 4>::size_type i{}; i < expected_adj_relations.size(); ++i)
    {
      REQUIRE(adj_relations[i] == expected_adj_relations[i]);
    }
  } 

  SECTION("1-to-2 flip (edge 02)")
  {
    loc_t const loc_code      = 10;
    auto  const adj_relations = set_interal_fract_adjacencies(fract_locs, loc_code);
 
    array<adjacency, 4> expected_adj_relations = {
      adjacency{ -1,   -1,   -1, -1 },
      adjacency{ -1,   -1, 1003, -1 },
      adjacency{ -1,   -1,   -1, -1 },
      adjacency{ -1, 1001,   -1, -1 }
    };

    for (
      typename array<adjacency, 4>::size_type i{}; i < expected_adj_relations.size(); ++i)
    {
      REQUIRE(adj_relations[i] == expected_adj_relations[i]);
    }
  } 

  SECTION("1-to-2 flip (edge 03)")
  {
    loc_t const loc_code      = 6;
    auto  const adj_relations = set_interal_fract_adjacencies(fract_locs, loc_code);
 
    array<adjacency, 4> expected_adj_relations = {
      adjacency{ -1,   -1,   -1, -1 },
      adjacency{ -1, 1002,   -1, -1 },
      adjacency{ -1,   -1, 1001, -1 },
      adjacency{ -1,   -1,   -1, -1 }
    };

    for (
      typename array<adjacency, 4>::size_type i{}; i < expected_adj_relations.size(); ++i)
    {
      REQUIRE(adj_relations[i] == expected_adj_relations[i]);
    }
  } 

  SECTION("1-to-2 flip (edge 12)")
  {
    loc_t const loc_code      = 9;
    auto  const adj_relations = set_interal_fract_adjacencies(fract_locs, loc_code);
 
    array<adjacency, 4> expected_adj_relations = {
      adjacency{ 1003, -1, -1, -1 },
      adjacency{   -1, -1, -1, -1 },
      adjacency{   -1, -1, -1, -1 },
      adjacency{ 1000, -1, -1, -1 }
    };

    for (
      typename array<adjacency, 4>::size_type i{}; i < expected_adj_relations.size(); ++i)
    {
      REQUIRE(adj_relations[i] == expected_adj_relations[i]);
    }
  } 

  SECTION("1-to-2 flip (edge 13)")
  {
    loc_t const loc_code      = 5;
    auto  const adj_relations = set_interal_fract_adjacencies(fract_locs, loc_code);
 
    array<adjacency, 4> expected_adj_relations = {
      adjacency{   -1, 1002, -1, -1 },
      adjacency{   -1,   -1, -1, -1 },
      adjacency{ 1000,   -1, -1, -1 },
      adjacency{   -1,   -1, -1, -1 }
    };

    for (
      typename array<adjacency, 4>::size_type i{}; i < expected_adj_relations.size(); ++i)
    {
      REQUIRE(adj_relations[i] == expected_adj_relations[i]);
    }
  } 

  SECTION("1-to-2 flip (edge 23)")
  {
    loc_t const loc_code      = 3;
    auto  const adj_relations = set_interal_fract_adjacencies(fract_locs, loc_code);
 
    array<adjacency, 4> expected_adj_relations = {
      adjacency{   -1, -1, 1001, -1 },
      adjacency{ 1000, -1,   -1, -1 },
      adjacency{   -1, -1,   -1, -1 },
      adjacency{   -1, -1,   -1, -1 }
    };

    for (
      typename array<adjacency, 4>::size_type i{}; i < expected_adj_relations.size(); ++i)
    {
      REQUIRE(adj_relations[i] == expected_adj_relations[i]);
    }
  } 

  // Vertex tests

  SECTION("1-to-1 flip (vertx 0)")
  {
    loc_t const loc_code      = 1;
    auto  const adj_relations = set_interal_fract_adjacencies(fract_locs, loc_code);
 
    array<adjacency, 4> expected_adj_relations = {
      adjacency{ -1, -1, -1, -1 },
      adjacency{ -1, -1, -1, -1 },
      adjacency{ -1, -1, -1, -1 },
      adjacency{ -1, -1, -1, -1 }
    };

    for (
      typename array<adjacency, 4>::size_type i{}; i < expected_adj_relations.size(); ++i)
    {
      REQUIRE(adj_relations[i] == expected_adj_relations[i]);
    }
  }

  SECTION("1-to-1 flip (vertx 1)")
  {
    loc_t const loc_code      = 2;
    auto  const adj_relations = set_interal_fract_adjacencies(fract_locs, loc_code);
 
    array<adjacency, 4> expected_adj_relations = {
      adjacency{ -1, -1, -1, -1 },
      adjacency{ -1, -1, -1, -1 },
      adjacency{ -1, -1, -1, -1 },
      adjacency{ -1, -1, -1, -1 }
    };

    for (
      typename array<adjacency, 4>::size_type i{}; i < expected_adj_relations.size(); ++i)
    {
      REQUIRE(adj_relations[i] == expected_adj_relations[i]);
    }
  }

  SECTION("1-to-1 flip (vertx 2)")
  {
    loc_t const loc_code      = 4;
    auto  const adj_relations = set_interal_fract_adjacencies(fract_locs, loc_code);
 
    array<adjacency, 4> expected_adj_relations = {
      adjacency{ -1, -1, -1, -1 },
      adjacency{ -1, -1, -1, -1 },
      adjacency{ -1, -1, -1, -1 },
      adjacency{ -1, -1, -1, -1 }
    };

    for (
      typename array<adjacency, 4>::size_type i{}; i < expected_adj_relations.size(); ++i)
    {
      REQUIRE(adj_relations[i] == expected_adj_relations[i]);
    }
  }

  SECTION("1-to-1 flip (vertx 3)")
  {
    loc_t const loc_code      = 8;
    auto  const adj_relations = set_interal_fract_adjacencies(fract_locs, loc_code);
 
    array<adjacency, 4> expected_adj_relations = {
      adjacency{ -1, -1, -1, -1 },
      adjacency{ -1, -1, -1, -1 },
      adjacency{ -1, -1, -1, -1 },
      adjacency{ -1, -1, -1, -1 }
    };

    for (
      typename array<adjacency, 4>::size_type i{}; i < expected_adj_relations.size(); ++i)
    {
      REQUIRE(adj_relations[i] == expected_adj_relations[i]);
    }
  }
}