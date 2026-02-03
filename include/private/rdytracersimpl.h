#ifndef RDYTRACERSIMPL_H
#define RDYTRACERSIMPL_H

#include <ceed/types.h>
#include <petscsys.h>
#include <private/rdycoreimpl.h>
#include <private/rdyoperatorimpl.h>

PETSC_INTERN PetscErrorCode CreateTracersQFunctionContext(Ceed, const RDyConfig, CeedQFunctionContext *);
PETSC_INTERN PetscErrorCode CreateTracersPetscInteriorFluxOperator(RDyMesh *, const RDyConfig, OperatorDiagnostics *, PetscOperator *);
PETSC_INTERN PetscErrorCode CreateTracersPetscBoundaryFluxOperator(RDyMesh *, const RDyConfig, RDyBoundary, RDyCondition, Vec, Vec,
                                                                    OperatorDiagnostics *, PetscOperator *);
PETSC_INTERN PetscErrorCode CreateTracersPetscSourceOperator(RDyMesh *, const RDyConfig, Vec, Vec, PetscOperator *);

#endif
