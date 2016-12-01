#include "catch.hpp"
#include "math/matrix.hpp"

TEST_CASE("Our matrix implementation")
{
  SECTION("should be comparable and accessible")
  {
    float const a = 1.0;
    float const b = 2.0;
    float const c = 3.0;
    float const d = 4.0;
    float const e = 5.0;
    float const f = 6.0;
    
    // aggregate constructor
    matrix<float, 2, 3> m{ { a, b, c, d, e, f } };
    
    // test for size() function
    REQUIRE(6 == m.size());
        
    // make sure that operator== works
    REQUIRE((m == matrix<float, 2, 3>{ a, b, c, d, e, f }));
        
    matrix<float, 2, 3> const not_m{ a, b, c, d, e, 7.0 };
        
    REQUIRE(m != not_m);
          
    // then our row() and col() functions should operate
    // as we expect
    REQUIRE((m.row(0) == vector<float, 3>{ a, b, c }));
    REQUIRE((m.row(1) == vector<float, 3>{ d, e, f }));
    
    REQUIRE((m.col(0) == vector<float, 2>{ a, d }));
    REQUIRE((m.col(1) == vector<float, 2>{ b, e }));
    REQUIRE((m.col(2) == vector<float, 2>{ c, f }));
  }

  SECTION("should support multiplication")
  {
    matrix<float, 2, 3> const a{ 1.0f, 2.0f, 3.0f,
                                 4.0f, 5.0f, 6.0f };
                                   
    matrix<float, 3, 2> const b{ 7.0f, 8.0f,
                                 9.0f, 10.0f,
                                 11.0f, 12.0f };
                                 
    matrix<float, 2, 2> const c{  58.0f,  64.0f,
                                 139.0f, 154.0f };
                                 
    REQUIRE(c == (a * b));
  }

  SECTION("should support determinant operations")
  {
    matrix<float, 4, 4> t{ 1.0f, 0.0f, 0.0f, 0.0f,
                           1.0f, 9.0f, 0.0f, 0.0f,
                           1.0f, 0.0f, 9.0f, 0.0f,
                           1.0f, 0.0f, 0.0f, 9.0f };
                           
    REQUIRE(eq(det(t), 729.0f));
    
    matrix<double, 4, 4> r{ 0.0, 1.85, 0.63, 2.65,
                           1.92, 1.57, 1.15, 2.94,
                            2.7, 2.45, 0.57, 2.81,
                           2.33, 1.68,  1.0, 0.05 };
           
    REQUIRE(eq(round_to(det(r), 3), -10.928));
    
    matrix<float, 4, 4> u{ 1.0, 0.0, 0.0, 0.0,
                           1.0, 9.0, 0.0, 0.0,
                           1.0, 0.0, 9.0, 0.0,
                           1.0, 3.0, 3.0, 0.0 };
                           
    REQUIRE(eq(det(u), 0.0f));
  }
}