/*
 * tetra.hpp
 *
 *  Created on: May 28, 2016
 *      Author: christian
 */

#ifndef TETRA_HPP_
#define TETRA_HPP_

// some convenience typedefs for easier refactoring in the future
typedef unsigned int integral;
typedef uint4 integral4;

size_t const num_vertices = 4;
size_t const num_neighbours = 4;

/*
 * We choose to separate out tetrahedral structure into two parts.
 * Namely, a 'tetra' which is just the vertices and then 'ngb' which
 * contains information about the adjacency relations between tetrahedra
 * in the mesh.
 */

struct tetra {
	integral v[num_vertices];
};

struct ngb {
	integral n[num_neighbours];
};

__device__
integral4 load_tetra(tetra const* tetrahedra, integral const pos);

__device__
integral4 load_ngb(ngb const* n, integral const pos);


#endif /* TETRA_HPP_ */
