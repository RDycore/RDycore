#ifndef RDYOPERATORIMPL_H
#define RDYOPERATORIMPL_H

#include <ceed/ceed.h>
#include <petsc/private/petscimpl.h>
#include <private/rdyboundaryimpl.h>
#include <private/rdyregionimpl.h>

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

// This type allows the direct manipulation of boundary values for the
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

PETSC_INTERN PetscErrorCode GetOperatorBoundaryData(RDy, RDyBoundary, OperatorBoundaryData *);
PETSC_INTERN PetscErrorCode SetOperatorBoundaryValues(OperatorBoundaryData *, PetscInt, PetscReal *);
PETSC_INTERN PetscErrorCode RestoreOperatorBoundaryData(RDy, RDyBoundary, OperatorBoundaryData *);

// This type allows the direct manipulation of source values on a specific
// region for the system of equations being solved by RDycore.
typedef struct {
  // associated RDy object
  RDy rdy;
  // associated region
  RDyRegion region;
  // number of components in the underlying system
  PetscInt num_components;
  // underlying data storage
  OperatorVectorData sources;
} OperatorSourceData;

PETSC_INTERN PetscErrorCode GetOperatorSourceData(RDy, RDyRegion, OperatorSourceData *);
PETSC_INTERN PetscErrorCode SetOperatorSourceValues(OperatorSourceData *, PetscInt, PetscReal *);
PETSC_INTERN PetscErrorCode GetOperatorSourceValues(OperatorSourceData *, PetscInt, PetscReal *);
PETSC_INTERN PetscErrorCode RestoreOperatorSourceData(RDy, RDyRegion, OperatorSourceData *);

// This type allows the direct manipulation of operator material properties on
// the entire domain for the system of equations being solved by RDycore.
typedef struct {
  // associated RDy object
  RDy rdy;
  // underlying data storage
  OperatorVectorData mannings;  // mannings coefficient
} OperatorMaterialData;

// operator material properties enum
typedef enum {
  OPERATOR_MANNINGS = 0,
} OperatorMaterialDataIndex;

PETSC_INTERN PetscErrorCode GetOperatorMaterialData(RDy, OperatorMaterialData *);
PETSC_INTERN PetscErrorCode SetOperatorMaterialValues(OperatorMaterialData *, OperatorMaterialDataIndex, PetscReal *);
PETSC_INTERN PetscErrorCode RestoreOperatorMaterialData(RDy, OperatorMaterialData *);

// This type allows the direct manipulation of flux divergences exchanged
// between the flux and source operators on the entire domain for the system of
// equations being solved by RDycore.
typedef struct {
  // associated RDy object
  RDy rdy;
  // number of components in the underlying system
  PetscInt num_components;
  // underlying data storage
  OperatorVectorData storage;
} OperatorFluxDivergenceData;

PETSC_INTERN PetscErrorCode GetOperatorFluxDivergenceData(RDy, OperatorFluxDivergenceData *);
PETSC_INTERN PetscErrorCode SetOperatorFluxDivergenceValues(OperatorFluxDivergenceData *, PetscInt, PetscReal *);
PETSC_INTERN PetscErrorCode RestoreOperatorFluxDivergenceData(RDy, OperatorFluxDivergenceData *);

#endif
