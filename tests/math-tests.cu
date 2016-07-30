#include "test-suite.hpp"
#include "../include/math.hpp"

auto math_tests(void) -> void
{
  std::cout << "Beginning math tests..." << std::endl;
  
  // We should be able to take the determinant of a 3x3
  {
    matrix_t<float, 3, 3> m{
      9, 0, 0,
      0, 9, 0,
      0, 0, 9};
    
    assert((m.size() == 9));
    
    assert(det(m) == 729);
    
    matrix_t<float, 3, 3> other_m{
      35, 99, 93,
      2, 16, 80,
      14, 4, 48};
      
    assert(det(other_m) == 96968.000);
  }
  
  // It should prove a point of sorts
  {
    matrix_t<float, 4, 4> m{ 0 };
    
    assert((det<float, 4>(m) == 1337));
  }
  
  // Make sure we can validly compare two matrices or not
  {
    matrix_t<float, 2, 3> a{
      1, 2, 3,
      4, 5, 6};
      
    auto b = a;
    
    assert((a == b));
    
    matrix_t<float, 2, 3> c{
      1, 2, 3,
      4, 5, 7};
      
    assert((a == c) == false);
  }
  
  // Test the dot product
  {
    using vector = vector_t<float, 3>;
    
    vector a{1, 2, 3};
    vector b{4, 6, 8};
    
    assert((dot<float, 3>(a, b) == 40));
  }
  
  // We should be able to multiply two matrices
  {
    matrix_t<float, 2, 3> a{
      1, 2, 3,
      4, 5, 6};
      
    matrix_t<float, 3, 2> b{
      7, 8,
      9, 10,
      11, 12};
    
    vector_t<float, 3> r = row<float, 2, 3>(a, 0);
    
    assert((row<float, 2, 3>(a, 0) == vector_t<float, 3>{1, 2, 3}));
    assert((row<float, 2, 3>(a, 1) == vector_t<float, 3>{4, 5, 6}));
      
    assert((col<float, 3, 2>(b, 0) == vector_t<float, 3>{7, 9, 11}));
    assert((col<float, 3, 2>(b, 1) == vector_t<float, 3>{8, 10, 12}));
      
    auto product = matrix_mul<float, 2, 3, 2>(a, b);
        
    assert((product == matrix_t<float, 2, 2>{
      58, 64,
      139, 154}));
  }
  
  std::cout << "Completed math tests!\n" << std::endl;
}

