#ifndef RDYOPERATORSIMPL_H
#define RDYOPERATORSIMPL_H

#include <ceed/ceed.h>
#include <petsc/private/petscimpl.h>
#include <private/rdycoreimpl.h>

// initialization/finalization functions
PETSC_INTERN PetscErrorCode InitOperators(RDy);
PETSC_INTERN PetscErrorCode DestroyOperators(RDy);

//----------------------
// Operator Data Access
//----------------------

// These types and functions allow access to data within operators, such as
// * boundary values (e.g. for Dirichlet boundary conditions)
// * source terms (water sources, momentum contributions)
// * relevant material properties (e.g. Mannings coefficient)
// * any needed intermediate quantities (e.g. flux divergences computed by the
//   flux operator and passed to the source operator)

// This type provides access to single- or multi-component vector data in either
// CEED or PETSc, depending upon whether CEED is enabled.
typedef struct {
  union {
    struct {
      CeedVector  vec;
      CeedScalar *data;
    } ceed;
    struct {
      Vec        vec;
      PetscReal *data;
    } petsc;
  };
  PetscBool updated;  // true iff updated
} OperatorVectorData;

// This type allows the direct manipulation of per-boundary values for the
// system of equations being solved by RDycore.
typedef struct {
  // associated RDy object
  RDy rdy;
  // associated boundary
  RDyBoundary boundary;
  // number of components in the underlying system
  PetscInt num_components;
  // underlying data storage
  OperatorVectorData storage;
} OperatorBoundaryData;

// This type allows the direct manipulation of source values on the entire
// domain for the system of equations being solved by RDycore.
typedef struct {
  // associated RDy object
  RDy rdy;
  // number of components in the underlying system
  PetscInt num_components;
  // underlying data storage
  OperatorVectorData sources;
  OperatorVectorData mannings;
  OperatorVectorData flux_divergence;
} OperatorSourceData;

PETSC_INTERN PetscErrorCode GetOperatorBoundaryData(RDy, RDyBoundary, OperatorBoundaryData *);
PETSC_INTERN PetscErrorCode SetOperatorBoundaryValues(OperatorBoundaryData *, PetscInt, PetscReal *);
PETSC_INTERN PetscErrorCode RestoreOperatorBoundaryData(RDy, OperatorBoundaryData *);

PETSC_INTERN PetscErrorCode GetOperatorSourceData(RDy, OperatorSourceData *);
PETSC_INTERN PetscErrorCode SetOperatorSourceValues(OperatorSourceData *, PetscInt, PetscReal *);
PETSC_INTERN PetscErrorCode RestoreOperatorSourceData(RDy, OperatorSourceData *);

#endif
