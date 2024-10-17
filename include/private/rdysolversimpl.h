#ifndef RDYSOLVERSIMPL_H
#define RDYSOLVERSIMPL_H

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

PETSC_INTERN PetscErrorCode InitSolvers(RDy);
PETSC_INTERN PetscErrorCode DestroySolvers(RDy);

PETSC_INTERN PetscErrorCode GetBoundaryData(RDy, PetscInt, BoundaryData *);
PETSC_INTERN PetscErrorCode SetBoundaryValues(BoundaryData, PetscInt, PetscReal *);
PETSC_INTERN PetscErrorCode CommitBoundaryValues(BoundaryData);

PETSC_INTERN PetscErrorCode GetSourceData(RDy, PetscInt, SourceData *);
PETSC_INTERN PetscErrorCode SetSourceValues(SourceData, PetscInt, PetscReal *);
PETSC_INTERN PetscErrorCode CommitSourceValues(SourceData);

#endif
