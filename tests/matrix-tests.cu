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
    matrix<float, 4, 4> a{ 11.0f,  9.0f, 24.0f, 2.0f,
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
    
    assert((L == decltype(L){     1.0f,     0.0f,    0.0f, 0.0f,
                              0.27273f,     1.0f,    0.0f, 0.0f,
                              0.09091f,  0.2875f,    1.0f, 0.0f,
                              0.18182f, 0.23125f, 0.0036f, 1.0f }));
                             

    assert((U == decltype(U){ 11.0f,      9.0f,     24.0f,     2.0f,
                               0.0f, 14.54545f, 11.45455f, 0.45455f,
                               0.0f,      0.0f,   -3.475f,  5.6875f,
                               0.0f,      0.0f,      0.0f, 0.51079f }));                             
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