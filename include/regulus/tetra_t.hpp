#ifndef REGULUS_TETRA_T_HPP_
#define REGULUS_TETRA_T_HPP_

#include "regulus/array.hpp"
#include "regulus/tetra_traits.hpp"

namespace regulus
{
  using tetra_t = array<ptrdiff_t, 4>;

  namespace traits
  {
    template <>
    struct coord_type<tetra_t>
    { using type = int; };

    template <>
    struct access<0, tetra_t>
    {
      static
      __host__ __device__
      auto get(tetra_t const& tetra) -> int
      { return tetra[0]; }

      static
      __host__ __device__
      void set(tetra_t& tetra, int const value)
      { tetra[0] = value; }
    };

    template <>
    struct access<1, tetra_t>
    {
      static
      __host__ __device__
      auto get(tetra_t const& tetra) -> int
      { return tetra[1]; }

      static
      __host__ __device__
      void set(tetra_t& tetra, int const value)
      { tetra[1] = value; }
    };

    template <>
    struct access<2, tetra_t>
    {
      static
      __host__ __device__
      auto get(tetra_t const& tetra) -> int
      { return tetra[2]; }

      static
      __host__ __device__
      void set(tetra_t& tetra, int const value)
      { tetra[2] = value; }
    };

    template <>
    struct access<3, tetra_t>
    {
      static
      __host__ __device__
      auto get(tetra_t const& tetra) -> int
      { return tetra[3]; }

      static
      __host__ __device__
      void set(tetra_t& tetra, int const value)
      { tetra[3] = value; }
    };
  }
}

#endif // REGULUS_TETRA_T_HPP_