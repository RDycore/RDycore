# RDycore (C) Style Guide

Here we provide a simple style guide for C code within the RDycore library. This
style guide applies mainly to source code within the RDycore C library itself.
The corresponding Fortran library is a thin hand-written wrapper around the C
library that doesn't leave much room for "style," so there is no Fortran style
guide.

This style guide is very short compared to many others, since C is a very small
language and most formatting conventions (whitespace, lengths of lines,
placement of curly braces) are enforced via tools and not humans. Here, we focus
mainly on those conventions that need the attention of RDycore team members and
other contributors.

## Style Conventions

RDycore's conventions are based on those in the
[PETSc Style and Usage Guide](https://petsc.org/release/developers/style/),
with the following deviations:

* we allow the use of variable-length arrays (VLAs)
* we use standard header guards instead of the nonstandard `#pragma once`
  mechanism
* we prefer C++-style code comments (`//`) to C-style comments (`/* */`) even
  for multi-line comments
* we use C conventions for logical expressions, e.g.
    * `if (cond)` instead of `if (cond == PETSC_TRUE)`
    * `if (!cond)` instead of `if (cond == PETSC_FALSE)`
    * `if (foo)` instead of `if (foo != NULL)` for pointers
    * `if (!foo)` instead of `if (foo == NULL)` for pointers

These deviations reflect certain facts about RDycore relative to PETSc:

* RDycore targets fewer systems than PETSc
* RDycore is much small codebase with fewer contributors than PETSc
* RDycore does not support the use of C++

We summarize the most important conventions below (in alphabetical order).

### Functions

Following the PETSc approach, all function bodies must be surrounded by
the `PetscFunctionBegin` and `PetscFunctionReturn` macros. Try to write short
functions, but avoid breaking up a function if doing so would introduce
undue complexity, such as passing several variables around or introducing
otherwise unnecessary abstractions.

Accordingly, almost all functions in RDycore return a `PetscErrorCode`
indicating success or failure, and all calls to these functions are enclosed
within the `PetscCall` macro, e.g.

```
  PetscCall(RDyAdvance(rdy));
```

Function inputs and outputs are passed as arguments to a function. Because C
passes arguments by value and not by reference (like Fortan), output parameters
are typically pointers.

### PETSc Data Types

We use data types defined by PETSc, which are configured according to RDycore's
build parameters:

* `PetscInt` is a 32-bit or 64-bit integer, depending on whether your PETSc
  installation is configured to use 32-bit or 64-bit indices. Use this for all
  integers **except** those transmitted via MPI (see below).
* `PetscMPIInt` is a 32-bit integer corresponding to the `MPI_INT` type, and
  should be used for all integers in MPI messages, even in 64-bit builds.
* `PetscReal` represents all real-valued quantities (usually double precision)
* `PetscBool` is PETSc's boolean type, which assumes one of the values
  `PETSC_TRUE` or `PETSC_FALSE`. It provides this type to provide compatibility
  with older revisions of the C language that don't include the standard C
  `bool` type.

### Header files

RDycore has one public header file that defines the interface for its C library:
`rdycore.h`. This file is generated by CMake, which injects configuration
information into the template `include/rdycore.h.in`. All other headers are
private and declare functions and types used internally within the RDycore
library.

Functions declared in `rdycore.h` are annotated with the `PETSC_EXTERN` macro,
which gives them external linkage, making them available to call outside of
RDycore. Functions in private headers are annotated instead with `PETSC_INTERN`,
giving them internal linkage and making them invisible to code outside of
RDycore.

All header files are located within the `include` directory. Private headers
live in the `private` subdirectory. Take a look at these header files to get a
sense of their structure.

### Naming

* **Variable names**: all variables (including fields within structs and
  function pointers) use `snake_case` (e.g. `water_height`, `num_cells`)
* **Function names**: all functions using `PascalCase` (e.g. `RDySetup`,
  `ApplyBoundaryCondition`)

In all cases, avoid inscrutible names that use single letters or unconventional
abbreviations unless their meaning is clear within the context in which they
appear.

### Structs

RDycore is developed using a "structured programming" approach in which related
data are grouped into `struct`s and passed to functions by value.

## Style Enforcement by Tools

To check and enforce the style in the code within your Git workspace against
the conventions we've adopted, you can use the following commands from your
build directory:

* `make format-c-check`: checks the source code in your workspace, reporting
  success or failure depending on whether it conforms to our style conventions
* `make format-c`: formats in place all relevant source code in your workspace,
  enforcing our style conventions

Sometimes an automated formatting tool disrupts the desired structure or layout
of a particular section of code. In this case, you can exclude the section from
auto-formatting by surrounding it with `// clang-format [on/off]` comments:

```
// clang-format on
<code section excluded from formatting>
// clang-format off
```

It's a good idea to run `make format-c` within a feature branch before creating
a pull request. This ensures that the automated formatting check will pass.
