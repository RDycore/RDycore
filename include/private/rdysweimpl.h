#ifndef RDYSWEIMPL_H
#define RDYSWEIMPL_H

#include <ceed/types.h>
#include <petscsys.h>
#include <private/rdycoreimpl.h>
#include <private/rdyoperatorimpl.h>

PETSC_INTERN PetscErrorCode CreateSWECeedInteriorFluxOperator(RDyMesh *, const RDyConfig, CeedOperator *);
PETSC_INTERN PetscErrorCode CreateSWECeedBoundaryFluxOperator(RDyMesh *, const RDyConfig, RDyBoundary, RDyCondition, CeedOperator *);
PETSC_INTERN PetscErrorCode CreateSWECeedSourceOperator(RDyMesh *, const RDyConfig, CeedVectorAndRestriction, CeedVectorAndRestriction,
                                                        CeedOperator *);

PETSC_INTERN PetscErrorCode CreateSWEPetscInteriorFluxOperator(RDyMesh *, const RDyConfig, OperatorDiagnostics *, PetscOperator *);
PETSC_INTERN PetscErrorCode CreateSWEPetscBoundaryFluxOperator(RDyMesh *, const RDyConfig, RDyBoundary, RDyCondition, Vec, Vec, OperatorDiagnostics *,
                                                               PetscOperator *);
PETSC_INTERN PetscErrorCode CreateSWEPetscSourceOperator(RDyMesh *, const RDyConfig, Vec, Vec, PetscOperator *);

#endif
