#include <cuda_profiler_api.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>

int main(void)
{
  cudaProfilerStart();     
  cudaProfilerStop();

  return 0;
}
