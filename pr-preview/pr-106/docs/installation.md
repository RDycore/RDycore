# Installation

RDycore builds and runs on the following platforms:

* Linux and Mac laptops and workstations
* [Frontier](https://www.olcf.ornl.gov/frontier/) (Oak Ridge National Laboratory)
* [Perlmutter](https://docs.nersc.gov/systems/perlmutter/) (NERSC)

## Required Software

To build RDycore, you need:

* [CMake v3.14+](https://cmake.org/)
* GNU Make
* reliable C, C++, and Fortran compilers
* a working MPI installation (like [OpenMPI](https://www.open-mpi.org/)
  or [MPICH](https://www.mpich.org/))
* [PETSc](https://petsc.org/release/) (Git revision `b3128c15e0e`), built with
  the following third-party libraries:
    * cgns
    * exodusii
    * fblaslapack
    * hdf5
    * libceed
    * metis
    * netcdf
    * parmetis
    * pnetcdf
    * zlib

You can obtain all of these freely on the Linux and Mac platforms. On Linux,
just use your favorite package manager. On a Mac, you can get the Clang C/C++
compiler by installing XCode, and then use a package manager like
[Homebrew](https://brew.sh/) or [MacPorts](https://www.macports.org/) to get
the rest.

## Clone the Repository

First, go get the [source code](https://github.com/RDycore/RDycore)
at GitHub:

=== "SSH"
    ```bash
    git clone git@github.com:RDycore/RDycore.git
    ```
=== "HTTPS"
    ```bash
    git clone https://github.com/RDycore/RDycore.git
    ```

This places an `RDycore` folder into your current path.

## Configure RDycore

RDycore uses CMake, and can be easily configured as long as
[PETSc is installed](https://petsc.org/release/install/) and [the `PETSC_DIR`
and `PETSC_ARCH` environment variables are set
properly](https://petsc.org/release/install/multibuild/#environmental-variables-petsc-dir-and-petsc-arch).
Usually all you need to do is change to your `RDycore` source directory and type

```bash
cmake -S . -B build
```

where `build` is the name of your build directory relative to the source
directory. If you want to install RDycore somewhere afterward, e.g. to be able
to configure E3SM to use it, you can set the prefix for the installation path
using the `CMAKE_INSTALL_PREFIX` parameter:

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/path/to/install
```

## Build, Test, and Install RDycore

After you've configured RDycore, you can build it:

1. From the build directory, type `make -j` to build the library.
4. To run tests for the library (and the included drivers), type
   `make test`.
5. To install the model to the location (indicated by your `CMAKE_INSTALL_PREFIX`,
   if you specified it), type `make install`. By default, products are installed
   in the `include`, `lib`, `bin`, and `share` subdirectories of this prefix.

## Making code changes and rebuilding

Notice that you must build RDycore in a  **build tree**, separate from its source
trees. This is standard practice in CMake-based build systems, and it allows you
to build several different configurations without leaving generated and compiled
files all over your source directory. However, you might have to change the way
you work in order to be productive in this kind of environment.

When you make a code change, make sure you build from the build directory that
you created in step 1 above:

```bash
cd /path/to/RDycore/build
make -j
```

You can also run tests from this build directory with `make test`.

This is very different from how some people like to work. One method of making
this easier is to use an editor in a dedicated window, and have another window
open with a terminal, sitting in your `build` directory. If you're using a fancy
modern editor, it might have a CMake-based workflow that handles all of this for
you.

The build directory has a structure that mirrors the source directory, and you
can type `make` in any one of its subdirectories to do partial builds. In
practice, though, it's safest to always build from the top of the build tree.

