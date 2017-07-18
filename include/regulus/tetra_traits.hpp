#ifndef REGULUS_TETRA_TRAITS_HPP_
#define REGULUS_TETRA_TRAITS_HPP_

namespace regulus
{
  namespace traits
  {
    template <size_t N, typename Tetra>
    struct access;

    template <typename Tetra>
    struct coord_type;
  }
}

#endif // REGULUS_TETRA_TRAITS_HPP_