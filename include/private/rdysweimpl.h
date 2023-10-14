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

PETSC_INTERN PetscErrorCode CreateSWEFluxOperator(Ceed ceed, RDyMesh *mesh, PetscInt num_boundaries, RDyBoundary boundaries[num_boundaries],
                                                  RDyCondition boundary_conditions[num_boundaries], PetscReal tiny_h, CeedOperator *flux_op);
PETSC_INTERN PetscErrorCode CreateSWESourceOperator(Ceed ceed, RDyMesh *mesh, RDyMaterial materials_by_cell[mesh->num_cells], PetscReal tiny_h,
                                                    CeedOperator *source_op);

PETSC_INTERN PetscErrorCode GetWaterSourceFromSWESourceOperator(CeedOperator source_op, CeedOperatorField *water_source_field);
PETSC_INTERN PetscErrorCode GetRiemannFluxFromSWESourceOperator(CeedOperator source_op, CeedOperatorField *riemann_flux_field);

PETSC_INTERN PetscErrorCode RiemannDataSWECreate(PetscInt N, RiemannDataSWE *data);
PETSC_INTERN PetscErrorCode RiemannDataSWEDestroy(RiemannDataSWE data);

#endif  // rdyswe_h
