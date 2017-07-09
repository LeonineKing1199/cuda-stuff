#ifndef REGULUS_POINT_HPP_
#define REGULUS_POINT_HPP_

#include <type_traits>

namespace regulus
{

template <
  typename T,
  typename = typename std::enable_if<std::is_arithmetic<T>::value>::type
>
struct point_t
{
  T x;
  T y;
  T z;
};

}

#endif // REGULUS_POINT_HPP_