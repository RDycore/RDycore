#ifndef RDYSEDIMENTIMPL_H
#define RDYSEDIMENTIMPL_H

#include <ceed/types.h>
#include <petscsys.h>
#include <private/rdycoreimpl.h>
#include <private/rdyoperatorimpl.h>

PETSC_INTERN PetscErrorCode CreateSedimentPetscInteriorFluxOperator(RDyMesh *, PetscInt, PetscInt, OperatorDiagnostics *, PetscReal, PetscOperator *);
PETSC_INTERN PetscErrorCode CreateSedimentPetscBoundaryFluxOperator(RDyMesh *, PetscInt, PetscInt, RDyBoundary, RDyCondition, Vec, Vec,
                                                                    OperatorDiagnostics *, PetscReal, PetscOperator *);
PETSC_INTERN PetscErrorCode CreateSedimentPetscSourceOperator(RDyMesh *, PetscInt, PetscInt, Vec, Vec, RDyFlowSourceMethod, PetscReal, PetscReal,
                                                              PetscOperator *);

#endif
