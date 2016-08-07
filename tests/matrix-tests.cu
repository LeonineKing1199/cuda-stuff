#include <stdio.h>

#include "test-suite.hpp"
#include "../include/math/matrix.hpp"

__host__ __device__
auto matrix_tests_impl(void) -> void
{
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
  
  // we should be able to create a diagonal matrix
  {
    matrix<float, 3, 3> const d = create_diagonal<float, 3>();
    
    assert((d == matrix<float, 3, 3>{ 1.0f, 0.0f, 0.0f,
                                      0.0f, 1.0f, 0.0f,
                                      0.0f, 0.0f, 1.0f }));
  }
  
  // we should be able to swap rows
  {
    matrix<float, 2, 3> A{ 1.0f, 2.0f, 3.0f,
                           4.0f, 5.0f, 6.0f };
                                 
    A.swap_rows(0, 1);
    
    assert((A == decltype(A){ 4.0f, 5.0f, 6.0f,
                              1.0f, 2.0f, 3.0f }));
  }
  
  // we should be able to create a pivot matrix
  {
    matrix<float, 3, 3> const a{ 1.0f, 3.0f, 5.0f,
                                 2.0f, 4.0f, 7.0f,
                                 1.0f, 1.0f, 0.0f };
                     
    auto p = pivot(a);
        
    assert((p == matrix<float, 3, 3>{ 0.0f, 1.0f, 0.0f,
                                      1.0f, 0.0f, 0.0f,
                                      0.0f, 0.0f, 1.0f }));
  }
  
  // we should be able to take the LU decomposition
  {
    matrix<float, 3, 3> const a{ 1.0f, 3.0f, 5.0f,
                                 2.0f, 4.0f, 7.0f,
                                 1.0f, 1.0f, 0.0f };
                                 
    matrix<float, 3, 3> L{ 0 };
    matrix<float, 3, 3> U{ 0 };
    
    LU_decompose<float, 3>(a, L, U);
    
    assert((L == matrix<float, 3, 3>{ 1.0f, 0.0f, 0.0f,
                                      0.5f, 1.0f, 0.0f,
                                      0.5f, -1.0f, 1.0f }));
                                      
    assert((U == matrix<float, 3, 3>{ 2.0f, 4.0f, 7.0f,
                                      0.0f, 1.0f, 1.5f,
                                      0.0f, 0.0f, -2.0f }));
  }
  
  // we should be able to take the LU decomposition (#2)
  {
    matrix<double, 4, 4> a{ 11.0f,  9.0f, 24.0f, 2.0f,
                            1.0f,  5.0f,  2.0f, 6.0f,
                            3.0f, 17.0f, 18.0f, 1.0f,
                            2.0f,  5.0f,  7.0f, 1.0f };
                            
    decltype(a) L{ 0 };
    decltype(a) U{ 0 };
    
    assert((pivot(a) == decltype(a){ 1.0f, 0.0f, 0.0f, 0.0f,
                                     0.0f, 0.0f, 1.0f, 0.0f,
                                     0.0f, 1.0f, 0.0f, 0.0f,
                                     0.0f, 0.0f, 0.0f, 1.0f }));
                                     
    LU_decompose(a, L, U);
    
    assert((L == decltype(L){ 1.0f, 0.0f, 0.0f, 0.0f,
                              0.272727272727273f, 1.0f, 0.0f, 0.0f,
                              0.090909090909091f, 0.287500000000000f, 1.0f, 0.0f,
                              0.181818181818182f, 0.231250000000000f, 0.003597122302158f, 1.0f }));
                             

    assert((U == decltype(U){ 11.0f,  9.0f, 24.0f,  2.0f,
                              0.0f, 14.545454545454500f, 11.454545454545500f,  0.454545454545455,
                              0.0f,  0.0f, -3.475000000000000,  5.687500000000000,
                              0.0f, 0.0f,  0.0f,  0.510791366906476f }));                             
  }
  
  // we should be able to take the determinant
  {
    matrix<float, 4, 4> t{ 1.0f, 0.0f, 0.0f, 0.0f,
                           1.0f, 9.0f, 0.0f, 0.0f,
                           1.0f, 0.0f, 9.0f, 0.0f,
                           1.0f, 0.0f, 0.0f, 9.0f };
                           
    assert(t.det() == 729);
    
    matrix<float, 4, 4> r{ 0.0, 1.85, 0.63, 2.65,
                           1.92, 1.57, 1.15, 2.94,
                           2.7, 2.45, 0.57, 2.81,
                           2.33, 1.68, 1.0, 0.05 };
                          
    assert(fabs(r.det() - -10.9277941) < (1e-6));
  }
}

__global__
void matrix_test_kernel(void)
{
  matrix_tests_impl();
}

auto matrix_tests(void) -> void
{
  std::cout << "Beginning matrix tests!" << std::endl;
     
  matrix_tests_impl();
  
  // this should also work on the device as well
  {
    matrix_test_kernel<<<1, 256>>>();
    cudaDeviceSynchronize();
  }
  
  std::cout << "Passed tests!\n" << std::endl;
}