#ifndef REGULUS_UTILS_NUMERIC_LIMITS_HPP_
#define REGULUS_UTILS_NUMERIC_LIMITS_HPP_

#include <cstdint>

namespace regulus
{
  template <typename T>
  struct numeric_limits
  {
    static
    constexpr
    __host__ __device__
    auto epsilon(void) -> T;

    static
    constexpr
    __host__ __device__
    auto max(void) -> T;
  };

  template <>
  struct numeric_limits<bool>
  {
    static
    constexpr
    __host__ __device__
    auto epsilon(void) -> bool
    {
      return false;
    }
  };

  template <>
  struct numeric_limits<char>
  {
    static
    constexpr
    __host__ __device__
    auto epsilon(void) -> char
    {
      return 0;
    }
  };

  template <>
  struct numeric_limits<signed char>
  {
    static
    constexpr
    __host__ __device__
    auto epsilon(void) -> signed char
    {
      return 0;
    }
  };

  template <>
  struct numeric_limits<unsigned char>
  {
    static
    constexpr
    __host__ __device__
    auto epsilon(void) -> unsigned char
    {
      return 0;
    }

    static
    constexpr
    __host__ __device__
    auto max(void) -> unsigned char
    {
      return UCHAR_MAX;
    }
  };

  template <>
  struct numeric_limits<wchar_t>
  {
    static
    constexpr
    __host__ __device__
    auto epsilon(void) -> wchar_t
    {
      return 0;
    }
  };

  template <>
  struct numeric_limits<char16_t>
  {
    static
    constexpr
    __host__ __device__
    auto epsilon(void) -> char16_t
    {
      return 0;
    }
  };

  template <>
  struct numeric_limits<char32_t>
  {
    static
    constexpr
    __host__ __device__
    auto epsilon(void) -> char32_t
    {
      return 0;
    }
  };

  template <>
  struct numeric_limits<short>
  {
    static
    constexpr
    __host__ __device__
    auto epsilon(void) -> short
    {
      return 0;
    }
  };

  template <>
  struct numeric_limits<unsigned short>
  {
    static
    constexpr
    __host__ __device__
    auto epsilon(void) -> unsigned short
    {
      return 0;
    }
  };

  template <>
  struct numeric_limits<int>
  {
    static
    constexpr
    __host__ __device__
    auto epsilon(void) -> int
    {
      return 0;
    }
  };

  template <>
  struct numeric_limits<unsigned int>
  {
    static
    constexpr
    __host__ __device__
    auto epsilon(void) -> unsigned int
    {
      return 0;
    }
  };

  template <>
  struct numeric_limits<long>
  {
    static
    constexpr
    __host__ __device__
    auto epsilon(void) -> long
    {
      return 0;
    }
  };

  template <>
  struct numeric_limits<unsigned long>
  {
    static
    constexpr
    __host__ __device__
    auto epsilon(void) -> unsigned long
    {
      return 0;
    }
  };

  template <>
  struct numeric_limits<long long>
  {
    static
    constexpr
    __host__ __device__
    auto epsilon(void) -> long long
    {
      return 0;
    }
  };

  template <>
  struct numeric_limits<unsigned long long>
  {
    static
    constexpr
    __host__ __device__
    auto epsilon(void) -> unsigned long long
    {
      return 0;
    }
  };

  template <>
  struct numeric_limits<float>
  {
    static
    constexpr
    __host__ __device__
    auto epsilon(void) -> float
    {
      return FLT_EPSILON;
    }
  };

  template <>
  struct numeric_limits<double>
  {
    static
    constexpr
    __host__ __device__
    auto epsilon(void) -> double
    {
      return DBL_EPSILON;
    }
  };

  template <>
  struct numeric_limits<long double>
  {
    static
    constexpr
    __host__ __device__
    auto epsilon(void) -> long double
    {
      return LDBL_EPSILON;
    }
  };
}

#endif // REGULUS_UTILS_NUMERIC_LIMITS_HPP_