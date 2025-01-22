#ifndef RDYSEDIMENTIMPL_H
#define RDYSEDIMENTIMPL_H

#include <ceed/types.h>
#include <petscsys.h>
#include <private/rdycoreimpl.h>
#include <private/rdyoperatorimpl.h>

PETSC_INTERN PetscErrorCode CreateSedimentCeedInteriorFluxOperator(RDyMesh *, const PetscInt, const PetscInt, const PetscReal, CeedOperator *);
PETSC_INTERN PetscErrorCode CreateSedimentCeedBoundaryFluxOperator(RDyMesh *, const PetscInt, const PetscInt, RDyBoundary, RDyCondition, PetscReal,
                                                                   CeedOperator *);
PETSC_INTERN PetscErrorCode CreateSedimentCeedSourceOperator(RDyMesh *, PetscInt, const PetscInt, RDyFlowSourceMethod, PetscReal, PetscReal,
                                                             CeedOperator *);
PETSC_INTERN PetscErrorCode CreateSedimentPetscInteriorFluxOperator(RDyMesh *, PetscInt, PetscInt, OperatorDiagnostics *, PetscReal, PetscOperator *);
PETSC_INTERN PetscErrorCode CreateSedimentPetscBoundaryFluxOperator(RDyMesh *, PetscInt, PetscInt, RDyBoundary, RDyCondition, Vec, Vec,
                                                                    OperatorDiagnostics *, PetscReal, PetscOperator *);
PETSC_INTERN PetscErrorCode CreateSedimentPetscSourceOperator(RDyMesh *, PetscInt, PetscInt, Vec, Vec, RDyFlowSourceMethod, PetscReal, PetscReal,
                                                              PetscOperator *);

#endif
