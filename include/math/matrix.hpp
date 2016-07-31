#ifndef REGULUS_MATRIX_HPP_
#define REGULUS_MATRIX_HPP_

#include <type_traits>
#include <array>
#include <utility>

#include "../common.hpp"

template <
  typename T,
  int N,
  int M,
  typename = typename reg::enable_if_t<std::is_floating_point<T>::value>
>
class matrix
{
private:
  std::array<T, N * M> data_;
  
public:
  template <typename ...Args>
  matrix(Args&&... args) : data_{std::forward<Args>(args)...}
  {}
  
  auto operator==(matrix<T, N, M> const& other) -> bool
  {
    bool equal = false;
    auto const& other_data = other.data_;
    
    for (int i = 0; i < data_.size(); ++i) {
      equal += (data_[i] != other_data[i] );
    }
      
    return !equal; 
  }
  
  auto operator!=(matrix<T, N, M> const& other) -> bool
  {
    return !(*this == other);
  }
};

#endif // REGULUS_MATRIX_HPP_