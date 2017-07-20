#ifndef REGULUS_ORIENTATION_HPP_
#define REGULUS_ORIENTATION_HPP_

#include <cstdint>

namespace regulus
{
  enum class orientation: uint8_t { positive = 1, zero = 0, negative = 2 };
}

#endif // REGULUS_ORIENTATION_HPP_