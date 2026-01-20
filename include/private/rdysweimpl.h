#ifndef RDYSWEIMPL_H
#define RDYSWEIMPL_H

#include <ceed/types.h>
#include <petscsys.h>
#include <private/rdycoreimpl.h>
#include <private/rdyoperatorimpl.h>

PETSC_INTERN PetscErrorCode CreateSWEQFunctionContext(Ceed, const RDyConfig, CeedQFunctionContext *);
PETSC_INTERN PetscErrorCode CreatePetscSWEInteriorFluxOperator(RDyMesh *, const RDyConfig, OperatorDiagnostics *, PetscOperator *);
PETSC_INTERN PetscErrorCode CreatePetscSWEBoundaryFluxOperator(RDyMesh *, const RDyConfig, RDyBoundary, RDyCondition, Vec, Vec, Vec,
                                                               OperatorDiagnostics *, PetscOperator *);
PETSC_INTERN PetscErrorCode CreatePetscSWESourceOperator(RDyMesh *, const RDyConfig, Vec, Vec, PetscOperator *);

#endif
