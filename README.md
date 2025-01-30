# RDycore

[![Build Status](https://github.com/RDycore/RDycore/workflows/auto_test/badge.svg)](https://github.com/RDycore/RDycore/actions)
[![Code Coverage](https://codecov.io/github/RDycore/RDycore/branch/main/graph/badge.svg?token=9RXZNKK194)](https://codecov.io/github/RDycore/RDycore)

[RDycore documentation](https://rdycore.github.io/RDycore/index.html)

## Required Software

To build RDycore, you need:

* A recent version of [CMake](https://cmake.org/)
* GNU Make
* reliable C and Fortran compilers
* a working MPI installation (like [OpenMPI](https://www.open-mpi.org/)
  or [Mpich](https://www.mpich.org/))
* A recent release or hash of [PETSc](https://gitlab.com/petsc/petsc)

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

## Configuring, Building, and Installing RDycore

Detailed instructions for getting started with RDycore, including required
versions of the above software, are [here](https://rdycore.github.io/RDycore/common/installation.html).
