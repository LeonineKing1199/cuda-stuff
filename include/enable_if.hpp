#ifndef REGULUS_ENABLE_IF_HPP_
#define REGULUS_ENABLE_IF_HPP_

#include <type_traits>

// helper template that emulates C++14 functionality
template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

#endif // REGULUS_ENABLE_IF_HPP_
