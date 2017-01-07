#ifndef REGULUS_INDEX_T_HPP_
#define REGULUS_INDEX_T_HPP_

#include "maybe-int.hpp"
using index_t = maybe_int<ptrdiff_t>;
using loc_t = maybe_int<char>;

#endif // REGULUS_INDEX_T_HPP_