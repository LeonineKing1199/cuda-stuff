#ifndef REGULUS_ALGORITHM_ORIENTATION_HPP_
#define REGULUS_ALGORITHM_ORIENTATION_HPP_

#include "regulus/is_point.hpp"

namespace regulus
{

enum class orientation { positive, zero, negative };

template <
  typename Point,
  typename = typename std::enable_if<is_point<Point>::value>::type>
__host__ __device__
auto orient(
  Point const a,
  Point const b,
  Point const c,
  Point const d) 
-> orientation
{
  
}

}


#endif // REGULUS_ALGORITHM_ORIENTATION_HPP_