#ifndef RDYSWEIMPL_H
#define RDYSWEIMPL_H

#include <ceed/types.h>
#include <petscsys.h>
#include <private/rdycoreimpl.h>
#include <private/rdyoperatorimpl.h>

PETSC_INTERN PetscErrorCode CreateSWECeedInteriorFluxOperator(RDyMesh *, PetscReal, CeedOperator *);
PETSC_INTERN PetscErrorCode CreateSWECeedBoundaryFluxOperator(RDyMesh *, RDyBoundary, RDyCondition, PetscReal, CeedOperator *);
PETSC_INTERN PetscErrorCode CreateSWECeedSourceOperator(RDyMesh *, PetscReal, CeedOperator *);

PETSC_INTERN PetscErrorCode CreateSWEPetscInteriorFluxOperator(RDyMesh *, OperatorDiagnostics *, PetscReal, PetscOperator *);
PETSC_INTERN PetscErrorCode CreateSWEPetscBoundaryFluxOperator(RDyMesh *, RDyBoundary, RDyCondition, Vec, Vec, OperatorDiagnostics *, PetscReal,
                                                               PetscOperator *);
PETSC_INTERN PetscErrorCode CreateSWEPetscSourceOperator(RDyMesh *, Vec, Vec, RDySourceTimeMethod, PetscReal, PetscReal, PetscOperator *);

#endif
