# CUDA Stuff

This is a much better attempt at degenerate Delaunay on the GPU.

I'm currently taking a sledge hammer to the project so it'll be unusable
for quite some time.

Currently, the build process is Windows-centric (all the various `.bat` files) but
they should be readily convertible to Bash scripts as well.

## Project Dependencies

This project requires CUDA 8.0 and above. It also relies on the 
[Catch testing framework](https://github.com/philsquared/Catch)
and [CMake](https://cmake.org/) 3.7 or higher.

## Building and Testing on Windows

For Windows, the only sane way of dealing with building CUDA is the [Ninja](https://ninja-build.org/) build tool
so you will need to download and install that as well. The reasons for this are, MSBuild doesn't extend well to the CUDA ecosystem (no parallel building, interface libs in CMake break header dependencies).

You can customize some of the top level parameters in the `vars.bat` file to control 
the CMake build process (configuration type, build directory, toolchain file location).
We recommend that you use [vcpkg](https://github.com/Microsoft/vcpkg) for your Windows 
development as it easily manages packages and generates a usable `CMAKE_TOOLCHAIN_FILE`.

Working with the code should be fairly simple. Open up an instance of the Developer Command 
Prompt for VS2015 and you can use `...\cuda-stuff>clean && build && test` to either purge
the currently built instance of the project, build the project or run the generated test binary
through `cuda-memcheck`.