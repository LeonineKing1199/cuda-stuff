#include <thrust/device_vector.h>

#include "test-suite.hpp"

#include "../include/globals.hpp"
#include "../include/domain.hpp"

#include "../include/lib/calc-ta-and-pa.hpp"
#include "../include/lib/nominate.hpp"
#include "../include/lib/fract-locations.hpp"
#include "../include/lib/fracture.hpp"
#include "../include/lib/redistribute-pts.hpp"
#include "../include/lib/get-assoc-size.hpp"

/*
  Kinda funny but these tests are basically an entire runthrough
  of the triangulation pipeline so this file makes a good reference
  for a rough approximation of how the triangulation routine really
  will work
*/

template <typename T>
__global__
void assert_givens(
  int const assoc_size,
  int const* __restrict__ pa,
  int const* __restrict__ ta,
  point_t<T>* const __restrict__ pts,
  tetra const* __restrict__ mesh)
{
  for (auto tid = get_tid(); tid < assoc_size; tid += grid_stride()) {
    point_t<T> const p = pts[pa[tid]];
    tetra const t = mesh[ta[tid]];
    
    point_t<T> const a = pts[t.x];
    point_t<T> const b = pts[t.y];
    point_t<T> const c = pts[t.z];
    point_t<T> const d = pts[t.w];
    
    assert(loc<T>(a, b, c, d, p) != -1);
  }
}

template <typename T>
__global__
void assert_redistributed_assocs(
  point_t<T> const* __restrict__ pts,
  tetra const* __restrict__ mesh,
  int const assoc_size,
  int const* __restrict__ pa,
  int const* __restrict__ ta,
  int const* __restrict__ la,
  int const* __restrict__ nm)
{
  for (auto tid = get_tid(); tid < assoc_size; tid += grid_stride()) {
    int const pa_id = pa[tid];
    int const ta_id = ta[tid];
    int const la_id = la[tid];
    
    assert(pa_id != -1);
    assert(ta_id != -1);
    assert(la_id != -1);
    
    assert(nm[pa_id] == 0);
    
    tetra const t = mesh[ta_id];
    
    auto const a = pts[t.x];
    auto const b = pts[t.y];
    auto const c = pts[t.z];
    auto const d = pts[t.w];
    
    assert((orient<T>(a, b, c, d) == orientation::positive));
    
    auto const p = pts[pa_id];
    
    assert((loc<T>(a, b, c, d, p) == la_id));
  }
}

auto redistribute_pts_tests(void) -> void
{
  std::cout << "Beginning redistribution tests!" << std::endl;
  
  // We should be able to redistribute points to the a new
  // set of tetrahedra from a fracture
  {
    using real = float;
    using thrust::device_vector;
    
    // generate cartesian grid points
    int const grid_length{9};
    int const root_coord_val{3 * grid_length};
    
    device_vector<point_t<real>> pts;
    pts.reserve(grid_length * grid_length * grid_length + 4);
    
    pts = gen_cartesian_domain<real>(grid_length);

    int const num_pts{static_cast<int>(pts.size())};
    assert(num_pts == (grid_length * grid_length * grid_length));
    
    int const num_est_tetrahedra = 8 * num_pts;

    // create the points for the initial global tetrahedron,
    // adding them to the memory buffer
    point_t<real> const root_a{0, 0, 0};
    point_t<real> const root_b{root_coord_val, 0, 0};
    point_t<real> const root_c{0, root_coord_val, 0};
    point_t<real> const root_d{0, 0, root_coord_val};
    
    pts.push_back(root_a);
    pts.push_back(root_b);
    pts.push_back(root_c);
    pts.push_back(root_d);
    
    // create root tetrahedron
    int const pts_size{static_cast<int>(pts.size())};
    int const a{pts_size - 4};
    int const b{pts_size - 3};
    int const c{pts_size - 2};
    int const d{pts_size - 1};
    tetra const t{a, b, c, d};
    
    // initialize mesh
    device_vector<tetra> mesh;
    mesh.resize(num_est_tetrahedra);
    mesh[0] = t;
    
    int const num_tetra{1};
    
    // initialize the association arrays
    device_vector<int> pa{num_est_tetrahedra, -1};
    device_vector<int> ta{num_est_tetrahedra, -1};
    device_vector<int> la{num_est_tetrahedra, -1};
    device_vector<int> fl{num_est_tetrahedra, -1};
    
    assert(pa.size() == 8 * (grid_length * grid_length * grid_length));
    
    calc_initial_assoc<real><<<bpg, tpb>>>(
      pts.data().get(),
      num_pts,
      t,
      pa.data().get(),
      ta.data().get(),
      la.data().get());
    
    cudaDeviceSynchronize();
    
    int assoc_size{num_pts};

    for (int i = 0; i < assoc_size; ++i) {
      assert(ta[i] == 0);
      assert(pa[i] >= 0 && pa[i] < num_pts);
    }

    assert_givens<real><<<bpg, tpb>>>(
      assoc_size,
      pa.data().get(),
      ta.data().get(),
      pts.data().get(),
      mesh.data().get());
    
    cudaDeviceSynchronize();

    device_vector<int> nm{num_pts, 0};
    
    //nominate(assoc_size, pa, ta, la, nm);
    nm[num_pts / 2] = 1;
    fract_locations(assoc_size, pa, nm, la, fl);
    fracture(assoc_size, num_tetra, pa, ta, la, nm, fl, mesh);
    redistribute_pts<real>(assoc_size, num_tetra, mesh, pts, nm, fl, pa, ta, la);
  }
  
  std::cout << "All tests passed!\n" << std::endl;
}
