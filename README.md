# CUDA Stuff

This is a much better attempt at degenerate [Delaunay](https://en.wikipedia.org/wiki/Delaunay_triangulation) on the GPU.

I'm currently taking a sledge hammer to the project so it'll be unusable
for quite some time.

Currently, the build process is Windows-centric (all the various `.bat` files) but
they should be readily convertible to Bash scripts as well.

## Introduction

Tetrahedral meshing has many uses, most of which are in the realm of physics and engineering where they can be used to accurately represent solids or fluid flows.

They can also be used to generate images and can be refined to exhibit desirable properties like spatial proximity.

Delaunay refinement is an overall goal of the project but first the goal is a working tetrahedralization routine.

## Algorithm

A summary of the algorithm is as follows:

### Preliminaries

1. Store all points and tetrahedra in linear buffers

   Points are simple `{x, y, z}` types and tetrahedra are
   a collection of 4 vertices, each vertex representing an index into the point buffer.

1. Use incremental construction

   Points are inserted into the current mesh, fracturing
   individual tetrahedra into a set of children

   What makes this project unique is how the insertion
   algorithm handles the case where a point is on the edge
   or face of a tetrahedron. We handle this case directly
   in the insertion routine which adds notable complexity
   but eschews the usage of perturbative schemes found in
   the Simulation of Simplicity.

1. Represent uninserted points using 3 arrays:
   1. `pa`, point associations (which point in point buffer)
   2. `ta`, tetrahedral assocation (which tetrahedon in mesh)
   3. `la`, location association (how is the point `p`
   spatially related to the tetrahedron `t`)

### Pipeline for Triangulation

1. Nominate points for insertion (nomination of points is constrained by how many points share tetrahedra)
3. Insert all nominated points and write new tetrahedra to the mesh buffer
4. Calculate new assocation relations for all points effected by the round of insertion

A full pipeline example can be found in the `redistribute_pts_test.cu` test file. It contains several intermediate routines and allocations that would detract from the overall explanation of the algorithm.

### Immediate TODOs

* Add Delaunay refinement
* Use arbitrary-precision orienation/insphere routines

## Project Dependencies

This project requires CUDA 9.0 and above. It also relies on the
[Catch testing framework](https://github.com/philsquared/Catch)
and [CMake](https://cmake.org/) 3.8 or higher.

### Building and Testing on Windows

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

### VSCode and CMake Tools Support

I couldn't get Visual Studio 2017 to manage CUDA and CMake all at the same time so
I fell back to VSCode and installed the CMake Tools extension by vector-of-bool.

We can have the IDE mirror our normal CLI build process by updating our `./.vscode/settings.json`
with:
```
    "cmake.configureSettings": {
        "CMAKE_TOOLCHAIN_FILE": "/vcpkg/scripts/buildsystems/vcpkg.cmake"
    },
    "cmake.buildDirectory": "${workspaceRoot}/build_${buildType}",
    "cmake.generator": "Ninja",
    "cmake.environment": {
        "CC": "cl",
        "CXX": "cl"
    }
```