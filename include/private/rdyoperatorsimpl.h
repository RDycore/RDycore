#ifndef RDYOPERATORSIMPL_H
#define RDYOPERATORSIMPL_H

#include <ceed/ceed.h>
#include <petsc/private/petscimpl.h>
#include <private/rdycoreimpl.h>

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
} BoundaryData;

// This type allows the direct manipulation of source values for the system
// of equations being solved by RDycore.
typedef struct {
  // associated RDy object
  RDy rdy;
  // associated region
  RDyRegion region;
  // number of components in the underlying system
  PetscInt num_components;
  // underlying data storage
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
} SourceData;

PETSC_INTERN PetscErrorCode InitOperators(RDy);
PETSC_INTERN PetscErrorCode DestroyOperators(RDy);

PETSC_INTERN PetscErrorCode AcquireBoundaryData(RDy, RDyBoundary, BoundaryData *);
PETSC_INTERN PetscErrorCode SetBoundaryValues(BoundaryData, PetscInt, PetscReal *);
PETSC_INTERN PetscErrorCode ReleaseBoundaryData(BoundaryData *);

PETSC_INTERN PetscErrorCode AcquireSourceData(RDy, RDyRegion, SourceData *);
PETSC_INTERN PetscErrorCode SetSourceValues(SourceData, PetscInt, PetscReal *);
PETSC_INTERN PetscErrorCode ReleaseSourceData(SourceData *);

#endif
