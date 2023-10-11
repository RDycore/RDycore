#ifndef RDYSWEIMPL_H
#define RDYSWEIMPL_H

#include <petscsys.h>
#include <private/rdymemoryimpl.h>

typedef struct {
  PetscInt   N;            // number of data values
  PetscReal *h, *hu, *hv;  // prognostic SWE variables
  PetscReal *u, *v;        // diagnostic variables
} RiemannDataSWE;

// Diagnostic structure that captures information about the conditions under
// which the maximum courant number is encountered. If you change this struct,
// update the call to MPI_Type_create_struct in InitMPITypesAndOps below.
typedef struct {
  PetscReal max_courant_num;  // maximum courant number
  PetscInt  global_edge_id;   // edge at which the max courant number was encountered
  PetscInt  global_cell_id;   // cell in which the max courant number was encountered
} CourantNumberDiagnostics;

static PetscErrorCode RiemannDataSWECreate(PetscInt N, RiemannDataSWE *data) {
  PetscFunctionBegin;

  data->N = N;
  PetscCall(RDyAlloc(PetscReal, data->N, &data->h));
  PetscCall(RDyAlloc(PetscReal, data->N, &data->hu));
  PetscCall(RDyAlloc(PetscReal, data->N, &data->hv));
  PetscCall(RDyAlloc(PetscReal, data->N, &data->u));
  PetscCall(RDyAlloc(PetscReal, data->N, &data->v));

  PetscCall(RDyFill(PetscReal, data->N, data->h, 0.0));
  PetscCall(RDyFill(PetscReal, data->N, data->hu, 0.0));
  PetscCall(RDyFill(PetscReal, data->N, data->hv, 0.0));
  PetscCall(RDyFill(PetscReal, data->N, data->u, 0.0));
  PetscCall(RDyFill(PetscReal, data->N, data->v, 0.0));

  PetscFunctionReturn(0);
}

static PetscErrorCode RiemannDataSWEDestroy(RiemannDataSWE data) {
  PetscFunctionBegin;

  data.N = 0;
  PetscCall(RDyFree(data.h));
  PetscCall(RDyFree(data.hu));
  PetscCall(RDyFree(data.hv));
  PetscCall(RDyFree(data.u));
  PetscCall(RDyFree(data.v));

  PetscFunctionReturn(0);
}

#endif  // rdyswe_h