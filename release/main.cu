#include "domain.hpp"

#include <iostream>
using std::cout;

int main(void)
{  
  cout << "Demoing the mesher!\n";
  using real = float;
      
  int const gl = 150;
  
  thrust::host_vector<point_t<real>> pts{gen_cartesian_domain<real>(gl)};

  cout << "Sorting " << pts.size() << " points\n";

  sort_by_peanokey<real>(pts);
  cudaDeviceSynchronize();

  cout << "Completed!\n\n";

  return 0;
}
