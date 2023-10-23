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

// PETSc (non-CEED) Riemann solver data for SWE
typedef struct {
  RiemannDataSWE  datal_internal_edges, datar_internal_edges;
  RiemannDataSWE *datal_bnd_edges, *datar_bnd_edges;
  RiemannDataSWE  data_cells;
} PetscRiemannDataSWE;

// Diagnostic structure that captures information about the conditions under
// which the maximum courant number is encountered. If you change this struct,
// update the call to MPI_Type_create_struct in InitMPITypesAndOps below.
typedef struct {
  PetscReal max_courant_num;  // maximum courant number
  PetscInt  global_edge_id;   // edge at which the max courant number was encountered
  PetscInt  global_cell_id;   // cell in which the max courant number was encountered
} CourantNumberDiagnostics;

PETSC_INTERN PetscErrorCode CreateSWEFluxOperator(Ceed, RDyMesh *, PetscInt n, RDyBoundary[n], RDyCondition[n], PetscReal, CeedOperator *);
PETSC_INTERN PetscErrorCode SWEFluxOperatorGetDirichletBoundaryValues(CeedOperator, RDyBoundary, CeedOperatorField *);
PETSC_INTERN PetscErrorCode SWEFluxOperatorSetDirichletBoundaryValues(CeedOperator, RDyBoundary boundary, PetscReal[boundary.num_edges]);

PETSC_INTERN PetscErrorCode CreateSWESourceOperator(Ceed, RDyMesh *mesh, RDyMaterial[mesh->num_cells], PetscReal, CeedOperator *);
PETSC_INTERN PetscErrorCode SWESourceOperatorGetWaterSource(CeedOperator, CeedOperatorField *);
PETSC_INTERN PetscErrorCode SWESourceOperatorGetRiemannFlux(CeedOperator, CeedOperatorField *);
PETSC_INTERN PetscErrorCode SWESourceOperatorSetWaterSource(CeedOperator, PetscReal *);

PETSC_INTERN PetscErrorCode RiemannDataSWECreate(PetscInt, RiemannDataSWE *);
PETSC_INTERN PetscErrorCode RiemannDataSWEDestroy(RiemannDataSWE);

PETSC_INTERN PetscErrorCode CreatePetscSWEFlux(PetscInt num_internal_edges, PetscInt n, RDyBoundary[n], void **);
PETSC_INTERN PetscErrorCode CreatePetscSWESource(RDyMesh *, void *);
PETSC_INTERN PetscErrorCode InitPetscSWEBoundaryFlux(void *, RDyCells *, RDyEdges *, PetscInt n, RDyBoundary[n], RDyCondition[n], PetscReal);
PETSC_INTERN PetscErrorCode GetPetscSWEDirichletBoundaryValues(void *, PetscInt, RiemannDataSWE *);

#endif
