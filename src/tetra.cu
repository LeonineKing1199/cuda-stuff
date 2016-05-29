#include "../include/tetra.hpp"

__device__
integral4 load_tetra(tetra const* tet, integral const pos) {
	return reinterpret_cast<integral4 const*>(tet)[pos];
}

__device__
integral4 load_ngb(ngb const* n, integral const pos) {
	return reinterpret_cast<integral4 const*>(n)[pos];
}
