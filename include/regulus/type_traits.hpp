#ifndef REGULUS_TYPE_TRAITS_HPP_
#define REGULUS_TYPE_TRAITS_HPP_

#include <type_traits>

namespace regulus
{
  template <bool B, typename T = void>
  using enable_if_t = typename std::enable_if<B, T>::type;
}

#endif // REGULUS_TYPE_TRAITS_HPP_