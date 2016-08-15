#include <stdio.h>

#include "test-suite.hpp"
#include "../include/math/matrix.hpp"

__host__ __device__
auto matrix_tests_impl(void) -> void
{
  // we should be able to construct a matrix type
  {
    float a = 1.0;
    float b = 2.0;
    float c = 3.0;
    float d = 4.0;
    float e = 5.0;
    float f = 6.0;
    
    matrix<float, 2, 3> m{ a, b, c, d, e, f };
    
    assert(m.data.size() == 6);
    assert((m == matrix<float, 2, 3>{ a, b, c, d, e, f }));
    
    matrix<float, 2, 3> const not_m{ a, b, c, d, e, 7.0 };
        
    assert((m != not_m));
    
    assert((m.row(0) == vector<float, 3>{ a, b, c }));
    assert((m.row(1) == vector<float, 3>{ d, e, f }));
    
    assert((m.col(0) == vector<float, 2>{ a, d }));
    assert((m.col(1) == vector<float, 2>{ b, e }));
    assert((m.col(2) == vector<float, 2>{ c, f }));
  }
  
  // we should be able to compare double types as well
  {
    vector<double, 4> x{ 1.0/7, 2.0/3, 3.0/4, 5.0/8 };
    for (int i = 0; i < 4; ++i) {
      x[i] = round_to(x[i], 9);
    }
    assert((x == decltype(x){ 0.142857143,
                              0.666666667,
                              0.75,
                              0.625 }));
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
    
    assert((pivot(a) == decltype(a){ 1, 0, 0, 0,
                                     0, 0, 1, 0,
                                     0, 1, 0, 0,
                                     0, 0, 0, 1 }));
                                     
    LU_decompose(a, L, U);
    
    for (int i = 0; i < 16; ++i) {
      L[i] = round_to(L[i], 5);
      U[i] = round_to(U[i], 5);
    }
    
    
    assert((L == decltype(L){ 1.0, 0.0, 0.0, 0.0,
                              0.27273, 1.0, 0.0, 0.0,
                              0.09091, 0.28750, 1.0, 0.0,
                              0.18182, 0.23125, 0.00360, 1.0 }));
                             

    assert((U == decltype(U){ 11.0, 9.0, 24.0, 2.0,
                              0.0, 14.54545, 11.45455, 0.45455,
                              0.0, 0.0, -3.47500, 5.68750,
                              0.0, 0.0, 0.0, 0.51079 }));                             
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
           
    assert(eq<float>(round_to<float>(r.det(), 3), -10.928));
    
    matrix<float, 4, 4> u{ 1.0, 0.0, 0.0, 0.0,
                           1.0, 9.0, 0.0, 0.0,
                           1.0, 0.0, 9.0, 0.0,
                           1.0, 3.0, 3.0, 0.0 };
                           
    assert(eq<float>(u.det(), 0.0));
  }
  
  // we shouldn't get weird undefined behavior
  { 
    matrix<float, 4, 4> const A{ 1, 0, 0, 0,
                                 1, 4.5, 0, 0,
                                 1, 0, 9, 0,
                                 1, 0, 0, 9 };
                                 
    auto const P = pivot(A);
    
    matrix<float, 4, 4> L;
    matrix<float, 4, 4> U;
                           
    LU_decompose(A, P, L, U);
    
    assert((L == decltype(A){ 1, 0, 0, 0,
                              1, 1, 0, 0,
                              1, 0, 1, 0,
                              1, 0, 0, 1 }));
                             
    assert((U == decltype(A){ 1, 0, 0, 0,
                              0, 4.5, 0, 0,
                              0, 0, 9, 0,
                              0, 0, 0, 9 }));
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