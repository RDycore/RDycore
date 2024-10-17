#include <ceed/ceed.h>
#include <petscdmceed.h>
#include <private/rdyoperatorsimpl.h>
#include <private/rdysweimpl.h>

// CEED uses C99 VLA features for shaping multidimensional
// arrays, which don't have the same drawbacks as VLA allocations.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla"

// these are used to register CEED operator events for profiling
static PetscClassId RDY_CLASSID;
PetscLogEvent       RDY_CeedOperatorApply;

// initializes operators for physics specified in the input configuration
PetscErrorCode InitOperators(RDy rdy) {
  PetscFunctionBegin;

  // register a logging event for applying our CEED operator
  PetscCall(PetscClassIdRegister("RDycore", &RDY_CLASSID));
  PetscCall(PetscLogEventRegister("CeedOperatorApp", RDY_CLASSID, &RDY_CeedOperatorApply));

  // just pass the call along for now
  InitSWE(rdy);

  PetscFunctionReturn(PETSC_SUCCESS);
}

// frees all resources devoted to operators
PetscErrorCode DestroyOperators(RDy rdy) {
  PetscFunctionBegin;

  // is anyone manipulating boundary data, source data, etc?
  if (rdy->lock.boundary_data) {
    for (PetscInt i = 0; i < rdy->num_boundaries; ++i) {
      PetscCheck(rdy->lock.boundary_data[i] == NULL, rdy->comm, PETSC_ERR_USER, "Could not destroy RDycore: boundary data is in use");
    }
  }
  if (rdy->lock.source_data) {
    for (PetscInt i = 0; i < rdy->num_regions; ++i) {
      PetscCheck(rdy->lock.source_data[i] == NULL, rdy->comm, PETSC_ERR_USER, "Could not destroy RDycore: source data is in use");
    }
  }
  PetscFree(rdy->lock.boundary_data);
  PetscFree(rdy->lock.source_data);

  PetscBool ceed_enabled = CeedEnabled(rdy);

  if (rdy->petsc.context) {
    PetscCall(DestroyPetscSWEFlux(rdy->petsc.context, ceed_enabled, rdy->num_boundaries));
  }

  if (ceed_enabled) {
    PetscCallCEED(CeedOperatorDestroy(&rdy->ceed.flux_operator));
    PetscCallCEED(CeedOperatorDestroy(&rdy->ceed.source_operator));
    PetscCallCEED(CeedVectorDestroy(&rdy->ceed.u_local));
    PetscCallCEED(CeedVectorDestroy(&rdy->ceed.rhs));
    PetscCallCEED(CeedVectorDestroy(&rdy->ceed.sources));
    if (rdy->ceed.host_fluxes) VecDestroy(&rdy->ceed.host_fluxes);
    // the CEED context belongs to RDycore itself, so we don't delete it
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// acquires exclusive access to boundary data for flux operators
PetscErrorCode AcquireBoundaryData(RDy rdy, RDyBoundary boundary, BoundaryData *boundary_data) {
  PetscFunctionBegin;
  if (!rdy->lock.boundary_data) {
    PetscCall(PetscCalloc1(rdy->num_boundaries, &rdy->lock.boundary_data));
  }
  PetscCheck(boundary.index >= 0 && boundary.index < rdy->num_boundaries, rdy->comm, PETSC_ERR_USER,
             "Invalid boundary for boundary data (index: %" PetscInt_FMT ")", boundary.index);
  PetscCheck(!rdy->lock.boundary_data[boundary.index], rdy->comm, PETSC_ERR_USER,
             "Could not acquire lock on boundary data -- another entity has access");
  rdy->lock.boundary_data[boundary.index] = boundary_data;
  boundary_data->rdy                      = rdy;
  boundary_data->boundary                 = boundary;
  PetscCall(VecGetBlockSize(rdy->u_global, &boundary_data->num_components));

  if (CeedEnabled(rdy)) {
    // get the relevant boundary sub-operator
    CeedOperator *sub_ops;
    PetscCallCEED(CeedCompositeOperatorGetSubList(rdy->ceed.flux_operator, &sub_ops));
    CeedOperator flux_op = sub_ops[1 + boundary_data->boundary.index];

    // fetch the array storing the boundary values
    CeedOperatorField field;
    PetscCallCEED(CeedOperatorGetFieldByName(flux_op, "q_dirichlet", &field));
    CeedVector vec;
    PetscCallCEED(CeedOperatorFieldGetVector(field, &vec));
    PetscCallCEED(CeedVectorGetArray(vec, CEED_MEM_HOST, &boundary_data->ceed.data));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// sets boundary values for the given component on the boundary associated with
// the given boundary data
PetscErrorCode SetBoundaryValues(BoundaryData boundary_data, PetscInt component, PetscReal *boundary_values) {
  PetscFunctionBegin;

  if (CeedEnabled(boundary_data.rdy)) {
    CeedScalar(*dirichlet_values)[boundary_data.num_components];
    *((CeedScalar **)&dirichlet_values) = boundary_data.ceed.data;

    // copy the boundary values into place
    for (CeedInt i = 0; i < boundary_data.boundary.num_edges; ++i) {
      dirichlet_values[i][component] = boundary_values[i];
    }
  } else {
    // FIXME: PETSc stuff goes here
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ReleaseBoundaryData(BoundaryData *boundary_data) {
  PetscFunctionBegin;
  if (CeedEnabled(boundary_data->rdy)) {
    // send the boundary values to the device
    PetscCallCEED(CeedVectorRestoreArray(boundary_data->ceed.vec, &boundary_data->ceed.data));
  } else {
    // FIXME: PETSc stuff goes here
  }
  boundary_data->rdy->lock.boundary_data[boundary_data->boundary.index] = NULL;
  *boundary_data                                                        = (BoundaryData){0};
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AcquireSourceData(RDy rdy, RDyRegion region, SourceData *source_data) {
  PetscFunctionBegin;

  if (!rdy->lock.boundary_data) {
    PetscCall(PetscCalloc1(rdy->num_boundaries, &rdy->lock.boundary_data));
  }
  PetscCheck(region.index >= 0 && region.index < rdy->num_regions, rdy->comm, PETSC_ERR_USER,
             "Invalid region for source data (index: %" PetscInt_FMT ")", region.index);
  PetscCheck(!rdy->lock.source_data[region.index], rdy->comm, PETSC_ERR_USER, "Could not acquire lock on source data -- another entity has access");
  rdy->lock.source_data[region.index] = source_data;
  source_data->rdy                    = rdy;
  source_data->region                 = rdy->regions[region.index];
  PetscCall(VecGetBlockSize(rdy->u_global, &source_data->num_components));

  if (CeedEnabled(rdy)) {
    // get the relevant source sub-operator
    CeedOperator *sub_ops;
    PetscCallCEED(CeedCompositeOperatorGetSubList(rdy->ceed.flux_operator, &sub_ops));
    CeedOperator source_op = sub_ops[0];

    // fetch the array storing the source values
    // FIXME: this is only valid for SWE
    CeedOperatorField field;
    PetscCallCEED(CeedOperatorGetFieldByName(source_op, "swe_src", &field));
    CeedVector vec;
    PetscCallCEED(CeedOperatorFieldGetVector(field, &vec));
    PetscCallCEED(CeedVectorGetArray(vec, CEED_MEM_HOST, &source_data->ceed.data));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// sets source values for the given component in the region associated with the
// given source data
PetscErrorCode SetSourceValues(SourceData source_data, PetscInt component, PetscReal *source_values) {
  PetscFunctionBegin;

  if (CeedEnabled(source_data.rdy)) {
    CeedScalar(*values)[source_data.num_components];
    *((CeedScalar **)&values) = source_data.ceed.data;

    // copy the boundary values into place
    for (CeedInt i = 0; i < source_data.region.num_cells; ++i) {
      values[i][component] = source_values[i];
    }
  } else {
    // FIXME: PETSc stuff goes here
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ReleaseSourceData(SourceData *source_data) {
  PetscFunctionBegin;
  if (CeedEnabled(source_data->rdy)) {
    // send the boundary values to the device
    PetscCallCEED(CeedVectorRestoreArray(source_data->ceed.vec, &source_data->ceed.data));
  } else {
    // FIXME: PETSc stuff goes here
  }
  source_data->rdy->lock.source_data[source_data->region.index] = NULL;
  *source_data                                                  = (SourceData){0};
  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
