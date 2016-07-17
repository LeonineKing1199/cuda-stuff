#ifndef REGULUS_TIMER_HPP_
#define REGULUS_TIMER_HPP_

#include <ctime>
#include <chrono>

/**
  I have little to no idea how this code actually works.
  I largely copy-pasted code from cppreference.com
*/

using namespace std::chrono;

class timer
{
private:
  
  high_resolution_clock::time_point start_;
  high_resolution_clock::time_point end_;
  
public:
  
  auto start(void) -> void
  {
    start_ = high_resolution_clock::now();
  }
  
  auto end(void) -> void
  {
    end_ = high_resolution_clock::now();
  }
  
  auto get_duration(void) -> double
  {
    return duration_cast<duration<double>>(end_ - start_).count();
  }
};

#endif // REGULUS_TIMER_HPP_
