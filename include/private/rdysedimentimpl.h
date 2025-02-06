#ifndef RDYSEDIMENTIMPL_H
#define RDYSEDIMENTIMPL_H

#include <ceed/types.h>
#include <petscsys.h>
#include <private/rdycoreimpl.h>
#include <private/rdyoperatorimpl.h>

PETSC_INTERN PetscErrorCode CreateSedimentCeedInteriorFluxOperator(RDyMesh *, const RDyConfig, CeedOperator *);
PETSC_INTERN PetscErrorCode CreateSedimentCeedBoundaryFluxOperator(RDyMesh *, const RDyConfig, RDyBoundary, RDyCondition, CeedOperator *);
PETSC_INTERN PetscErrorCode CreateSedimentCeedSourceOperator(RDyMesh *, const RDyConfig, CeedVectorAndRestriction, CeedVectorAndRestriction,
                                                             CeedOperator *);
PETSC_INTERN PetscErrorCode CreateSedimentPetscInteriorFluxOperator(RDyMesh *, const RDyConfig, OperatorDiagnostics *, PetscOperator *);
PETSC_INTERN PetscErrorCode CreateSedimentPetscBoundaryFluxOperator(RDyMesh *, const RDyConfig, RDyBoundary, RDyCondition, Vec, Vec,
                                                                    OperatorDiagnostics *, PetscOperator *);
PETSC_INTERN PetscErrorCode CreateSedimentPetscSourceOperator(RDyMesh *, const RDyConfig, Vec, Vec, PetscOperator *);

#endif
