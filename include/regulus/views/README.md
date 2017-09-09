### Motivation

Creating a custom view interpretation and then subsequent abstraction
seems foolhardy at best and it largely is so this is a toy view
lib that's intended to only support this project.

The inspiration comes from string_view in C++17 and parts of
MS' array_view type as well.

Design Goals:
* Some form of container type erasure (via pointers and size)
* Cheap to copy by value
* Supported equally well on the host and device
* Easy to pass to kernels
* Able to work with Thrust

### Classes

`array_view`: mutable data, cannot be re-aliased, non-resizable
`const_array_view`: immutable data, can be re-aliased, non-resizable