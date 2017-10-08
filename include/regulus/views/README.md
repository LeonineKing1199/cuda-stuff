### Motivation

Design Goals:
* Some form of container type erasure (via pointers and size)
* Cheap to copy by value
* Supported equally well on the host and device
* Easy to pass to kernels
* Able to work with Thrust seamlessly-ish (user needs execution policies)

Creating a custom view interpretation and then subsequent abstraction
seems foolhardy at best and it largely is so this is a toy view
lib that's intended to only support this project.

The inspiration comes from string_view in C++17 and the span
types that can be found in the GSL and Google's Abseil (mostly
Abseil). Unlike the GSL span, compile-time spans are not optimized
in size down to a single pointer.

Parts of the code are direct translations of the Abseil code into
my style so most of the credit belongs to the Abseil people and their
implementation.