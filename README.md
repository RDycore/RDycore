# RDycore

## Required Software

To build RDycore, you need:

* [CMake v3.12+](https://cmake.org/)
* GNU Make
* reliable C and Fortran compilers
* a working MPI installation (like [OpenMPI](https://www.open-mpi.org/)
  or [Mpich](https://www.mpich.org/))

You can obtain all of these freely (except perhaps your favorite Fortran
compiler) on the Linux and Mac platforms. On Linux, just use your favorite
package manager. On a Mac, you can get the Clang C/C++ compiler by installing
XCode, and then use a package manager like
[Homebrew](https://brew.sh/) or [MacPorts](https://www.macports.org/) to get the
rest.

For example, to download relevant software on your Mac using Homebrew, type

```
brew install cmake gfortran openmpi
```

## Building RDycore

To configure, build, and install RDycore:

1. Make sure you have the latest versions of all required Git submodules:
   ```
   git submodule update --init --recursive
   ```
2. Make sure you set your `PETSC_DIR` and `PETSC_ARCH` environment variables
   to refer to an existing installation of PETSc (v3.18.1 or later).
3. Create a directory in which you'll build RDycore (a "build directory").
   For example:
   ```
   mkdir build
   ```
4. Change to your build directory and use the `cmake` command to configure your
   build (we describe available CMake options below). For example:
   ```
   cd build
   cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install -DCMAKE_BUILD_TYPE=Debug
   ```
5. If your configuration process above succeeded, you can build RDycore
   from your build directory with `make`:
   ```
   make -j
   ```
6. To run tests for RDycore, type
   ```
   make test
   ```
7. To install RDycore to your desired location (unnecessary if you're just doing
   development work), type
   ```
   make install
   ```

### Supported configuration options

CMake allows you to specify build options with the `-D` flag, as indicated in
Step 3 above. Here are the options supported by RDycore:

* **`CMAKE_INSTALL_PREFIX=/path/to/install`**: a path to which the RDycore library
  and driver are installed with `make install` (as in Step 7)
* **`CMAKE_BUILD_TYPE=Debug|Release`**: controls whether a build has debugging
  information or whether it is optimized

Since RDycore gets most of its configuration information from PETSc, we don't
need to use most other CMake options.

### Making code changes and rebuilding

This project uses **build trees** that are separate from source trees. This
is standard practice in CMake-based build systems, and it allows you to build
several different configurations without leaving generated and compiled files
all over your source directory. However, you might have to change the way you
work in order to be productive in this kind of environment.

When you make a code change, make sure you build from the build directory that
you created in step 1 above:

```
cd /path/to/RDycore/build
make -j
```

You can also run tests from this build directory with `make test` (described
below).

This is very different from how some people like to work. One method of making
this easier is to use an editor in a dedicated window, and have another window
open with a terminal, sitting in your `build` directory.

The build directory has a structure that mirrors the source directory, and you
can type `make` in any one of its subdirectories to do partial builds. In
practice, though, it's safest to always build from the top of the build tree.

## Running Tests

RDycore uses [CTest](https://cmake.org/cmake/help/book/mastering-cmake/chapter/Testing%20With%20CMake%20and%20CTest.html),
CMake's testing program, to run its tests. CTest is very fancy and allows us to
run tests selectively and in various ways, but all you need to do to run all the
tests for RDycore is to change to your build directory and type

```
make test
```

This runs every test defined in your build configuration and dumps the results
to `Testing/Temporary/LastTest.log`.
