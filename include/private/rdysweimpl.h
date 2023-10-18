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
PETSC_INTERN PetscErrorCode CreateSWESourceOperator(Ceed, RDyMesh *m, RDyMaterial[m->num_cells], PetscReal, CeedOperator *);

PETSC_INTERN PetscErrorCode GetWaterSourceFromSWESourceOperator(CeedOperator, CeedOperatorField *);
PETSC_INTERN PetscErrorCode GetRiemannFluxFromSWESourceOperator(CeedOperator, CeedOperatorField *);

PETSC_INTERN PetscErrorCode RiemannDataSWECreate(PetscInt, RiemannDataSWE *);
PETSC_INTERN PetscErrorCode RiemannDataSWEDestroy(RiemannDataSWE);

PETSC_INTERN PetscErrorCode CreatePetscSWEFlux(PetscInt num_internal_edges, PetscInt n, RDyBoundary[n], void **);
PETSC_INTERN PetscErrorCode CreatePetscSWESource(RDyMesh *, void *);
PETSC_INTERN PetscErrorCode InitBoundaryPetscSWEFlux(RDyCells *, RDyEdges *, PetscInt n, RDyBoundary[n], RDyCondition[n], PetscReal, void **);

#endif  // rdyswe_h
