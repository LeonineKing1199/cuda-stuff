#ifndef REGULUS_DEVICE_VECTOR_FORWARD_DECLARATION_HPP_
#define REGULUS_DEVICE_VECTOR_FORWARD_DECLARATION_HPP_

// I know this is a bad practice
// If Thrust changes their template signature,
// this code'll be broken. Good news is, if this
// code ever fails for that reason then the whole
// project can be repaired relatively simply by
// updating the template signature in every forward
// declaration
namespace thrust {
  template <typename T, typename Alloc>
  class device_vector;
}

#endif // REGULUS_DEVICE_VECTOR_FORWARD_DECLARATION_HPP_
