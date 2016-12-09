#ifndef REGULUS_SIZE_TYPE_HPP_
#define REGULUS_SIZE_TYPE_HPP_

#include "enable_if.hpp"

// This may be overengineering...
// The idea is that by using an ecapsulated
// class typedef, we can restrict the allowed
// types to prevent non-breaking refactors
// So just make sure that size_type is always
// assigned to the encapsulated typedef

template <
  typename T,
  typename = enable_if_t<std::is_signed<T>::value>,
  typename = enable_if_t<std::is_integral<T>::value>
>
struct signed_type_container
{
  using type = T;
};

template <
  typename T,
  typename = enable_if_t<std::is_unsigned<T>::value>,
  typename = enable_if_t<std::is_integral<T>::value>
>
struct unsigned_type_container
{
  using type = T;
};

using int_type = long long int;
using uint_type = unsigned long long int;

// Create some global, project-wide size type configuration
// We use long long because GPUs finally have more than 4 GB
// of RAM now!
using size_type = typename signed_type_container<int_type>::type;
using usize_type = typename unsigned_type_container<uint_type>::type;

#endif // REGULUS_SIZE_TYPE_HPP_