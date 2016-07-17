#include "test-suite.hpp"
#include "../include/mesher.hpp"

auto mesher_tests(void) -> void
{
  // We should be able to construct a mesh
  {
    using real = float;
    
    mesher<real> m;
  }
}