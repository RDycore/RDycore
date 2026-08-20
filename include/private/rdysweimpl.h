#ifndef RDYSWEIMPL_H
#define RDYSWEIMPL_H

#include <ceed/types.h>
#include <petscsys.h>
#include <private/rdycoreimpl.h>
#include <private/rdyoperatorimpl.h>

PETSC_INTERN PetscErrorCode CreateSWEQFunctionContext(Ceed, const RDyConfig, CeedQFunctionContext *);
PETSC_INTERN PetscErrorCode CreatePetscSWEInteriorFluxOperator(RDyMesh *, MPI_Comm, const RDyConfig, OperatorDiagnostics *, PetscOperator *);
PETSC_INTERN PetscErrorCode CreatePetscSWEInteriorFluxHROperator(RDyMesh *, const RDyConfig, OperatorDiagnostics *, PetscOperator *);
PETSC_INTERN PetscErrorCode CreatePetscSWEBoundaryFluxOperator(RDyMesh *, const RDyConfig, RDyBoundary, RDyCondition, Vec, Vec, Vec,
                                                               OperatorDiagnostics *, PetscOperator *);
PETSC_INTERN PetscErrorCode CreatePetscSWESourceOperator(RDyMesh *, const RDyConfig, Vec, Vec, PetscOperator *);
PETSC_INTERN PetscErrorCode CreatePetscSWESourceHROperator(RDyMesh *, const RDyConfig, Vec, Vec, PetscOperator *);

// SWE RHS Jacobian (swe_jacobian_petsc.c): matrix creation and TS registration
PETSC_INTERN PetscErrorCode RegisterSWERHSJacobian(RDy);
PETSC_INTERN PetscErrorCode RegisterSWEIMEXFriction(RDy);
PETSC_INTERN PetscErrorCode DestroySWERHSJacobian(RDy);

#endif
