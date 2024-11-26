#ifndef RDYSWEIMPL_H
#define RDYSWEIMPL_H

#include <ceed/types.h>
#include <petscsys.h>
#include <private/rdyoperatorimpl.h>

PETSC_INTERN PetscErrorCode CreateSWECeedInteriorFluxOperator(RDyMesh *, PetscReal, CeedOperator *);
PETSC_INTERN PetscErrorCode CreateSWECeedBoundaryFluxOperator(RDyMesh *, RDyBoundary, PetscReal, CeedOperator *);
PETSC_INTERN PetscErrorCode CreateSWECeedExternalSourceOperator(RDyMesh *, RDyRegion, PetscReal, CeedOperator *);

PETSC_INTERN PetscErrorCode CreateSWEPetscInteriorFluxOperator(RDyMesh *, PetscReal, PetscOperator *);
PETSC_INTERN PetscErrorCode CreateSWEPetscBoundaryFluxOperator(RDyMesh *, RDyBoundary, Vec, PetscReal, PetscOperator *);
PETSC_INTERN PetscErrorCode CreateSWEPetscExternalSourceOperator(RDyMesh *, RDyRegion, Vec, PetscReal, PetscOperator *);

// FIXME: vvv old stuff vvv

/*
PETSC_INTERN PetscErrorCode RiemannDataSWECreate(PetscInt, RiemannDataSWE *);
PETSC_INTERN PetscErrorCode RiemannDataSWEDestroy(RiemannDataSWE);
PETSC_INTERN PetscErrorCode RiemannEdgeDataSWECreate(PetscInt, PetscInt, RiemannEdgeDataSWE *);
PETSC_INTERN PetscErrorCode RiemannEdgeDataSWEDestroy(RiemannEdgeDataSWE);

PETSC_INTERN PetscErrorCode CreatePetscSWEFluxForInternalEdges(RDyEdges *edges, PetscInt num_comp, PetscInt num_internal_edges, void **);
PETSC_INTERN PetscErrorCode CreatePetscSWEFluxForBoundaryEdges(RDyEdges *edges, PetscInt num_comp, PetscInt n, RDyBoundary *, PetscBool, void **);
PETSC_INTERN PetscErrorCode DestroyPetscSWEFlux(void *, PetscBool, PetscInt);
PETSC_INTERN PetscErrorCode CreatePetscSWESource(RDyMesh *, void *);
PETSC_INTERN PetscErrorCode InitPetscSWEBoundaryFlux(void *, RDyCells *, RDyEdges *, PetscInt n, RDyBoundary *, RDyCondition *, PetscReal);
PETSC_INTERN PetscErrorCode GetPetscSWEDirichletBoundaryValues(void *, PetscInt, RiemannDataSWE *);

PETSC_INTERN PetscErrorCode SWEFindMaxCourantNumber(RDy);
*/

#endif
