# RDycore Code Structure and Organization

In this section, we describe the organization of RDycore's source files. We have
attempted to group functions and data types together into subsystems that can
be easily understood separately from one another.

## A Brief Tour of RDycore's Source Tree

The root level of the [RDycore](https://github.com/RDycore/RDycore) source tree
contains a `README.md` file, a `CMakeLists.txt` file that defines the build
system, and some configuration files for documentation and tooling.

There are several folders at the root of the source tree:

```
+ RDyore +- [.github/workflows](https://github.com/RDycore/RDycore/tree/main/.github/workflows]
         |
         +- [cmake](https://github.com/RDycore/RDycore/tree/main/cmake)
         |
         +- [config](https://github.com/RDycore/RDycore/tree/main/config)
         |
         +- [docs](https://github.com/RDycore/RDycore/tree/main/docs)
         |
         +- [driver](https://github.com/RDycore/RDycore/tree/main/driver)
         |
         +- [external](https://github.com/RDycore/RDycore/tree/main/external)
         |
         +- [include](https://github.com/RDycore/RDycore/tree/main/include)
         |
         +- [share](https://github.com/RDycore/RDycore/tree/main/share)
         |
         +- [src](https://github.com/RDycore/RDycore/tree/main/src)
         |
         +- [tools](https://github.com/RDycore/RDycore/tree/main/src)
```

* `.github/workflows`: workflows that support our [GitHub Continuous Integration
  Environment](development.md#GitHub-Continuous-Integration-Environment)
* `cmake`: CMake scripts that support the build system, testing, etc.
* `config`: shell scripts to help with running RDycore on our target platforms
* `docs`: Markdown source files for our [mkdocs](https://squidfunk.github.io/mkdocs-material/)-based
  documentation
* `driver`: C and Fortran driver programs, including the standalone RDycore
  drivers and a few other development tools
* `external`: third-party libraries used by RDycore that aren't provided by
  PETSc
* `include`: the `rdycore.h` API header file, and all private header files
  (located within the `private` subfolder)
* `share`: data files for initial conditions, material properties, and
  unstructured grids (used mainly for testing)
* `src`: the source code for RDycore
* `tools`: miscellaneous tools and scripts for building and deploying Docker
  images

Take a look at each of these folders to familiarize yourself with their
contents. In particular, the `src` folder has a few important subfolders:

* `src/f90-mod`: the RDycore Fortran module that mirrors the C library's
  capabilities
* `src/swe`: functions and data structures specific to the solution of the
  2D shallow water equations
* `src/tests`: unit tests for subsystems within RDycore

The private headers in the `include` folder contain definitions of opaque types
and functions that are helpful throughout RDycore but are not part of the API.

The `driver` folder also has its own `tests` subfolder that defines tests
that run the standalone drivers and perform convergence tests for selected
problems.

## The RDycore Object and its Lifecycle

RDycore (the "river dynamical core") is represented in code by a data structure
called `RDy`, declared as an opaque type in [include/rdycore.h](https://github.com/RDycore/RDycore/tree/main/include/rdycore.h).
It is defined in [include/private/rdycoreimpl.h](https://github.com/RDycore/RDycore/tree/main/include/private/rdycoreimpl.h).
In this section, we describe how to manipulate `RDy` to set up and run
simulations. It may be helpful to refer to the [standalone C driver](https://github.com/RDycore/RDycore/tree/main/driver/main.c)
(and or the corresponding [Fortran driver](https://github.com/RDycore/RDycore/tree/main/driver/main.F90))
as you read this section.

1. **Initialization**

Before RDycore can be used in any program, the supporting systems must be
initialized with a call to `RDyInit`:

```
  // initialize RDycore subsystems
  PetscCall(RDyInit(argc, argv, helpString));
```

This function accepts the `argc` and `argv` command line argument parameters
accepted by any C program, plus a string printed to display help/usage
information.

Next, create an `RDy` object, passing it an MPI communicator, the path to a
[YAML input file](../common/input.md), and the pointer to the desired `RDy`
object:

```
  // create an RDy on the given communicator with the given input
  RDy rdy;
  PetscCall(RDyCreate(MPI_COMM_WORLD, "my-input.yaml", &rdy));
```

This only creates the object--it does not read input or allocate any resources
for the problem described within the input.

2. **Setup**

To read the input file and ready your `RDy` object to run the simulation
defined therein, call `RDySetup`:

```
  // read input and set up the dycore
  PetscCall(RDySetup(rdy));
```

After this call, you're ready to run a simulation. You can also all any query
or utility functions on your `RDy` object to do whatever you need for your own
simulation. For example, you might need information about the unstructured grid,
or perhaps you are specifying specific boundary values for a Dirichlet boundary
condition. See the API header file for all the possibilities.

3. **Timestepping**

The simplest way to start running your simulation after setup is to run
`RDyAdvance` (which takes a single step) within a `while` loop that uses
`RDyFinished` as termination condition:

```
  // run the simulation to completion
  while (!RDyFinished(rdy)) {
    // ... pre-step logic here ...

    // advance the simulation by a single step
    PetscCall(RDyAdvance(rdy));

    // ... post-step logic here ...
  }
```

If you look in the standalone C driver, you can see various pre- and post-step
logic to accommodate data transfer, restarts, etc.

4. **Finalization**

At the end of an RDycore-enabled program, you must destroy the `RDy` object with
a call to `RDyDestroy`:

```
  // destroy the dycore
  PetscCall(RDyDestroy(&rdy));
```

Finally, call `RDyFinalize` to reclaim all resources used by RDycore and its
subsystems:

```
  // clean up
  PetscCall(RDyFinalize());
```



## Computational Domain

### Regions

### Boundaries

## Operators

**NOTE**: at the time of writing, RDycore solves the shallow water equations
exclusively, so everything here refers to operators pertaining only to these
equations.

### Riemann Flux operator

### Source operator

## Input

## Output

### Visualization

### Time series

## Checkpoints and Restarts

## Running Ensembles

### Ensemble configuration process

### Differences between "ensemble mode" and "single-simulation mode"

## Support for Fortran

A Fortran version of the public C interface is available in an [`rdycore` Fortran module](https://github.com/RDycore/RDycore/blob/main/src/f90-mod/rdycore.F90)
in the `src/f90-mod` directory. This module is hand-written and uses the
[Fortran 2003 `iso_c_binding`](https://fortranwiki.org/fortran/show/iso_c_binding)
standard intrinsic module to map C functions to equivalent Fortran subroutines
and functions, and defines appropriate data types.

The mapping from a C function to a Fortran subroutine (or function) is
accomplished in two parts:

1. A C function is made available to Fortran by defining a Fortran function
   (or function) using a `bind(C)` annotation specifying the case-sensitive name
   of the C function. All data types in this Fortran function must correspond to
   supported C data types via the `iso_c_binding` module. The function returns
   an integer corresponding to the `PetscErrorCode` return type of the C
   function. We refer to such a function as a "C-bound Fortran function".

2. A Fortran "wrapper" subroutine is defined that calls the C-bound Fortran
   function defined in item 1. This subroutine follows the PETSc convention in
   which the last argument (`ierr`) gets the return value of the C-bound Fortran
   function.

Whenever a new C function is added to the public interface in `rdycore.h`, these
two corresponding Fortran items must be added to support its use in Fortran.

### Special considerations

* **C pointers**: Perhaps counterintuitively, C pointers must be passed by value
  to Fortran-bound C functions. In other words, any argument of type `type(c_ptr)`
  must have the `value` attribute and `intent(in)`. You can think of a pointer
  as a memory location, which is morally equivalent to an integer of the
  appropriate size. The `intent(in)` expresses that the pointer itself remains
  unchanged even if the data it points to is modified by the function.
  **NOTE**: an `RDy` C object is expressed as a C pointer in the Fortran module.
* **C primitives**: Because C passes arguments to functions by value and Fortran
  by reference, it is necessary to add the `value` attribute to any parameter
  in a C-bound Fortran function that has a non-pointer C type. This includes
  most `intent(in)` primitive parameters in C-bound Fortran functions. Note
  that `intent(out)` parameters in these functions must necessarily be pointers
  in C functions, so they must not have the `value` attribute.
* **Enumerated types**: An enumerated type in C can be mapped to Fortran as a
  set of related integer parameters. See, for example, the way [time units](https://github.com/RDycore/RDycore/blob/main/src/f90-mod/rdycore.F90#L35)
  are expressed in the `rdycore` Fortran module.
* **PETSc types**: PETSc types like `Vec` are passed to C-bound Fortran
  functions as `PetscFortranAddr` with the `value` attribute and `intent(in)`.
  This is very similar to the way we treat C pointers, with some magic PETSc
  pixie dust sprinkled on it to satisfy conditions required by PETSc.
