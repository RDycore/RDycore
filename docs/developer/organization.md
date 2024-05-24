# RDycore Code Structure and Organization

In this section, we describe the organization of RDycore's source files. We have
attempted to group functions and data types together into subsystems that can
be easily understood separately from one another.

## A Brief Tour of RDycore's Source Tree

## The RDycore Object and its Lifecycle

### Initialization

### Setup

### Timestepping

### Post-processing

### Finalization

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
