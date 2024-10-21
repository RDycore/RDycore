#ifndef RDYSWEIMPL_H
#define RDYSWEIMPL_H

#include <ceed/types.h>
#include <petscsys.h>
#include <private/rdycoreimpl.h>

typedef struct {
  PetscInt   N;            // number of data values
  PetscReal *h, *hu, *hv;  // prognostic SWE variables
  PetscReal *u, *v;        // diagnostic variables
} RiemannDataSWE;

typedef struct {
  PetscInt   N;        // number of data values
  PetscReal *cn, *sn;  // cosine and sine of the angle between the edge and y-axis
  PetscReal *flux;     // flux through the edge
  PetscReal *amax;     // courant number
} RiemannEdgeDataSWE;

// PETSc (non-CEED) Riemann solver data for SWE
typedef struct {
  RiemannDataSWE      datal_internal_edges, datar_internal_edges;
  RiemannEdgeDataSWE  data_internal_edges;
  RiemannDataSWE     *datal_bnd_edges, *datar_bnd_edges;
  RiemannEdgeDataSWE *data_bnd_edges;
  RiemannDataSWE      data_cells;
} PetscRiemannDataSWE;

PETSC_INTERN PetscErrorCode InitSWE(RDy);

PETSC_INTERN PetscErrorCode CreateSWEFluxOperator(Ceed, RDyMesh *, CeedInt n, RDyBoundary *, RDyCondition *, PetscReal, CeedOperator *);
PETSC_INTERN PetscErrorCode SWEFluxOperatorSetTimeStep(CeedOperator, PetscReal);
PETSC_INTERN PetscErrorCode SWEFluxOperatorGetBoundaryFlux(CeedOperator, RDyBoundary, CeedOperatorField *);

PETSC_INTERN PetscErrorCode CreateSWESourceOperator(Ceed, RDyMesh *mesh, PetscInt num_cells, RDyMaterial *, PetscReal, CeedOperator *);
PETSC_INTERN PetscErrorCode SWESourceOperatorSetTimeStep(CeedOperator, PetscReal);
PETSC_INTERN PetscErrorCode SWESourceOperatorGetRiemannFlux(CeedOperator, CeedOperatorField *);

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
#endif
