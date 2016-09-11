#include <thrust/device_vector.h>

#include "test-suite.hpp"

#include "../include/globals.hpp"
#include "../include/domain.hpp"

#include "../include/lib/calc-ta-and-pa.hpp"
#include "../include/lib/nominate.hpp"
#include "../include/lib/fract-locations.hpp"
#include "../include/lib/fracture.hpp"
#include "../include/lib/redistribute-pts.hpp"

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

auto redistribute_pts_tests(void) -> void
{
  std::cout << "Beginning redistribution tests!" << std::endl;
  
  // We should be able to redistribute points to the a new
  // set of tetrahedra from a fracture
  {
    using real = float;
    
    // generate cartesian grid points
    int const grid_length = 9;
    int const root_coord_val = 3 * grid_length;
    thrust::device_vector<point_t<real>> pts;
    pts.reserve(grid_length * grid_length * grid_length + 4);
    pts = gen_cartesian_domain<real>(grid_length);

    int const num_pts = pts.size();
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
    int const pts_size{(int ) pts.size()};
    int const a{pts_size - 4};
    int const b{pts_size - 3};
    int const c{pts_size - 2};
    int const d{pts_size - 1};
    tetra const t{a, b, c, d};
    
    // initialize mesh
    thrust::device_vector<tetra> mesh;
    mesh.reserve(num_est_tetrahedra);
    mesh[0] = t;
    
    // initialize the association arrays
    thrust::device_vector<int> pa{num_est_tetrahedra, -1};
    thrust::device_vector<int> ta{num_est_tetrahedra, -1};
    thrust::device_vector<int> la{num_est_tetrahedra, -1};
    thrust::device_vector<int> fl{num_est_tetrahedra, -1};
    
    assert(pa.capacity() == 8 * (grid_length * grid_length * grid_length));
    
    calc_initial_assoc<real><<<bpg, tpb>>>(
      pts.data().get(),
      num_pts,
      t,
      pa.data().get(),
      ta.data().get(),
      la.data().get());
    
    cudaDeviceSynchronize();
    
    int const assoc_size = num_pts;

    for (int i = 0; i < assoc_size; ++i) {
      assert(ta[i] == 0);
    }

    assert_givens<real><<<bpg, tpb>>>(
      assoc_size,
      pa.data().get(),
      ta.data().get(),
      pts.data().get(),
      mesh.data().get());
    
    cudaDeviceSynchronize();
    
    int const num_tetra{1};
    
    thrust::device_vector<int> nm{num_pts, 1};
    thrust::device_vector<int> nm_ta{num_tetra, -1};
    
    nominate<<<bpg, tpb>>>(
      assoc_size,
      ta.data().get(),
      pa.data().get(),
      nm_ta.data().get(),
      nm.data().get());
    
    repair_nm_ta<<<bpg, tpb>>>(
      assoc_size,
      pa.data().get(),
      ta.data().get(),
      nm.data().get(),
      nm_ta.data().get());
    
    cudaDeviceSynchronize();

    assert(nm[pa[nm_ta[0]]] == 1);

    thrust::device_vector<int> num_redistributions{1, 0};
    
    fract_locations(
      pa.data().get(),
      nm.data().get(),
      la.data().get(),
      assoc_size,
      fl.data().get());
        
    fracture<<<bpg, tpb>>>(
      assoc_size,
      num_tetra,
      pa.data().get(),
      ta.data().get(),
      la.data().get(),
      nm.data().get(),
      fl.data().get(),
      mesh.data().get());
    
    redistribute_pts<real><<<bpg, tpb>>>(
      assoc_size,
      num_tetra,
      mesh.data().get(),
      pts.data().get(),
      nm.data().get(),
      nm_ta.data().get(),
      fl.data().get(),
      pa.data().get(),
      ta.data().get(),
      la.data().get(),
      num_redistributions.data().get());
    
    cudaDeviceSynchronize();
    
    assert(num_redistributions[0] > 0);
    
    auto begin = thrust::make_zip_iterator(
      thrust::make_tuple(pa.begin(), ta.begin(), la.begin()));
    
    thrust::sort(begin, begin + pa.size(), [] __device__ (
      thrust::tuple<int, int, int> const&a,
      thrust::tuple<int, int, int> const&b) -> bool
    {      
      unsigned int const pa_a = reinterpret_cast<unsigned int const&>(thrust::get<0>(a));
      unsigned int const pa_b = reinterpret_cast<unsigned int const&>(thrust::get<0>(b));
      
      return pa_a < pa_b;
    });
    
    /*for (int i = 0; i < (int ) pa.size(); ++i) {
      if (la[i] == -1) {
        continue;
      }
      
      std::cout << i << " : " << pa[i] << " " << ta[i] << " " << la[i]
                << " ? " << nm[pa[i]] << std::endl;
    }*/
  }
  
  std::cout << "All tests passed!\n" << std::endl;
}
