#include "domain.hpp"

#include <cuda_profiler_api.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>

using std::cout;
using thrust::host_vector;
using thrust::device_vector;

int main(void)
{  
  cout << "Demoing the mesher!\n";
  using real = float;
  
  cudaProfilerStart();     
  int const gl = 150;
  
  device_vector<point_t<real>> pts = gen_cartesian_domain<real>(gl);

  cout << "Sorting " << pts.size() << " points\n";

  sort_by_peanokey<real>(pts);
  cudaDeviceSynchronize();
  cudaProfilerStop();

  cout << "Completed! Sorted " << pts.size() << " points!\n\n";

  return 0;
}
