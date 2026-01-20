#ifndef RDYTRACERIMPL_H
#define RDYTRACERIMPL_H

#include <ceed/types.h>
#include <petscsys.h>
#include <private/rdycoreimpl.h>
#include <private/rdyoperatorimpl.h>

PETSC_INTERN PetscErrorCode CreateTracerQFunctionContext(Ceed, const RDyConfig, CeedQFunctionContext *);
PETSC_INTERN PetscErrorCode CreatePetscTracerInteriorFluxOperator(RDyMesh *, const RDyConfig, OperatorDiagnostics *, PetscOperator *);
PETSC_INTERN PetscErrorCode CreatePetscTracerBoundaryFluxOperator(RDyMesh *, const RDyConfig, RDyBoundary, RDyCondition, Vec, Vec,
                                                                  OperatorDiagnostics *, PetscOperator *);
PETSC_INTERN PetscErrorCode CreatePetscTracerSourceOperator(RDyMesh *, const RDyConfig, Vec, Vec, PetscOperator *);

#endif
