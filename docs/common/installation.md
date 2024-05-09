# Installation

You can build and run RDycore on the following platforms:

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
* [PETSc](https://petsc.org/release/), built with the following third-party
  libraries:
    * cgns
    * exodusii
    * fblaslapack
    * hdf5
    * libceed
    * metis
    * muparser
    * netcdf
    * parmetis
    * pnetcdf
    * zlib

You can obtain all of these freely on the Linux and Mac platforms. On Linux,
just use your favorite package manager. On a Mac, you can get the Clang C/C++
compiler by installing XCode, and then use a package manager like
[Homebrew](https://brew.sh/) or [MacPorts](https://www.macports.org/) to get
the rest.

### Which version of PETSc?

Check [our automated testing workflow](https://github.com/RDycore/RDycore/blob/main/.github/workflows/auto_test.yml#L24)
for the proper Git hash to use to build RDycore. The linked line specifies a
Docker image containing the "blessed" version of PETSc, which can be read as
follows:

```
coherellc/rdycore-petsc:fc288817-int32
```

* `coherellc` is the name of the DockerHub organization hosting the image
* `rdycore-petsc` is the the name of the Docker image
* `fc288817` is the Git hash within the [PETSc repository](https://gitlab.com/petsc/petsc)
  used to build RDycore
* `int32` (or `int64`) indicates whether the PETSc installation within the image
  uses 32-bit or 64-bit integers for the `PetscInt` data type.

See our [PETSc Dockerfile](https://github.com/RDycore/RDycore/blob/main/tools/Dockerfile.petsc)
for the commands we use to build PETSc in our continous integration environment.

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


## Preinstalled PETSc for RDycore on certain DOE machines

The RDycore team supports installation of the model at following DOE machines:

1. [Perlmutter](https://docs.nersc.gov/systems/perlmutter/) at NERSC
2. [Frontier](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html) at OLCF
3. [Aurora](https://www.alcf.anl.gov/support-center/aurora/getting-started-aurora) at ALCF


First, run the shell script, `config/set_petsc_settings.sh`, to set PETSc-related environmental
variables for the pre-installed PETSc on these supported machines and load appropriate modules.
PETSc has been pre-installed with support for 32-bit and 64-bit integers on these supported machines,
and the 32-bit PETSc installation is the default setting. The 64-bit PETSc installation can be used
by passing the optional command line argument `--64bit` to `config/set_petsc_settings.sh`.

The Perlmutter system has two types of compute nodes: CPU-only and CPU-GPU nodes.
   The PETSc settings for CPU or GPU nodes can be selected via `--pm cpu` or `--pm gpu`, respectively.

```bash
source config/set_petsc_settings.sh --pm <cpu|gpu> <--64bit>

```

On all other systems, the script can be run as

```bash
source config/set_petsc_settings.sh <--64bit>

```

After setting PETSc variables and loading the appropriate modules, follow the
steps outlined in the [previous section](#build-test-and-install-rdycore) to build the code.