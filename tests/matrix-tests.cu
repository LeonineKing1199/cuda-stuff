#include "test-suite.hpp"
#include "../include/math/matrix.hpp"

/*__global__
void test_kernel(void)
{
  matrix<float, 2, 3> const a{ 1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f };
                               
  matrix<float, 3, 2> const b{ 7.0f, 8.0f,
                               9.0f, 10.0f,
                               11.0f, 12.0f };
                               
  matrix<float, 2, 2> const c{  58.0f,  64.0f,
                               139.0f, 154.0f };
                               
  assert((a * b == c));
}*/

auto matrix_tests(void) -> void
{
  std::cout << "Beginning matrix tests!" << std::endl;
  /*
  // we should be able to construct a matrix type
  {
    float a = 1;
    float b = 2;
    float c = 3;
    float d = 4;
    float e = 5;
    float f = 6;
    
    matrix<float, 2, 3> m{ a, b, c, d, e, f };
    
    assert((m == matrix<float, 2, 3>{ a, b, c, d, e, f }));
    assert((m != matrix<float, 2, 3>{ a, b, c, d, e, (float ) 7 }));
    
    assert((m.row(0) == vector<float, 3>{ a, b, c }));
    assert((m.row(1) == vector<float, 3>{ d, e, f }));
    
    assert((m.col(0) == vector<float, 2>{ a, d }));
    assert((m.col(1) == vector<float, 2>{ b, e }));
    assert((m.col(2) == vector<float, 2>{ c, f }));
  }
  
  // we should be able to take the dot product of two vectors
  {    
    vector<float, 3> const a{ 1.0f, 2.0f, 3.0f };
    vector<float, 3> const b{ 4.0f, 8.0f, 16.0f };
    
    assert((a * b == 68));
  }
  
  // we should be able to do matrix multiplication correctly
  {
    matrix<float, 2, 3> const a{ 1.0f, 2.0f, 3.0f,
                                 4.0f, 5.0f, 6.0f };
                                 
    matrix<float, 3, 2> const b{ 7.0f, 8.0f,
                                 9.0f, 10.0f,
                                 11.0f, 12.0f };
                                 
    matrix<float, 2, 2> const c{  58.0f,  64.0f,
                                 139.0f, 154.0f };
                                 
    assert((a * b == c));
  }
  
  // this should also work on the device as well
  {
    //test_kernel<<<1, 256>>>();
    //cudaDeviceSynchronize();
  }
  
  std::cout << "Passed tests!\n" << std::endl;*/
}