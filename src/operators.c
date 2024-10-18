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
  PetscCheck(rdy->lock.source_data == NULL, rdy->comm, PETSC_ERR_USER, "Could not destroy RDycore: source data is in use");
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
PetscErrorCode GetOperatorBoundaryData(RDy rdy, RDyBoundary boundary, OperatorBoundaryData *boundary_data) {
  PetscFunctionBegin;
  *boundary_data = (OperatorBoundaryData){0};
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

    // fetch the relevant vector
    CeedOperatorField field;
    PetscCallCEED(CeedOperatorGetFieldByName(flux_op, "q_dirichlet", &field));
    PetscCallCEED(CeedOperatorFieldGetVector(field, &boundary_data->storage.ceed.vec));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// sets boundary values for the given component on the boundary associated with
// the given operator boundary data
PetscErrorCode SetOperatorBoundaryValues(OperatorBoundaryData *boundary_data, PetscInt component, PetscReal *boundary_values) {
  PetscFunctionBegin;

  if (CeedEnabled(boundary_data->rdy)) {
    // if this is the first update, get access to the vector's data
    if (!boundary_data->storage.updated) {
      PetscCallCEED(CeedVectorGetArray(boundary_data->storage.ceed.vec, CEED_MEM_HOST, &boundary_data->storage.ceed.data));
      boundary_data->storage.updated = PETSC_TRUE;
    }

    // reshape for multicomponent access
    CeedScalar(*values)[boundary_data->num_components];
    *((CeedScalar **)&values) = boundary_data->storage.ceed.data;

    // set the data
    for (CeedInt e = 0; e < boundary_data->boundary.num_edges; ++e) {
      values[e][component] = boundary_values[e];
    }
  } else {
    // if this is the first update, get access to the vector's data
    if (!boundary_data->storage.updated) {
      PetscCall(VecGetArray(boundary_data->storage.petsc.vec, &boundary_data->storage.petsc.data));
      boundary_data->storage.updated = PETSC_TRUE;
    }

    // set the data
    PetscReal *u = boundary_data->storage.petsc.data;
    for (PetscInt e = 0; e < boundary_data->boundary.num_edges; ++e) {
      u[e * boundary_data->num_components + component] = boundary_values[e];
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RestoreOperatorBoundaryData(RDy rdy, OperatorBoundaryData *boundary_data) {
  PetscFunctionBegin;
  PetscCheck(rdy == boundary_data->rdy, rdy->comm, PETSC_ERR_USER, "Could not restore operator boundary data: wrong RDy");
  if (CeedEnabled(boundary_data->rdy)) {
    if (boundary_data->storage.updated) {
      PetscCallCEED(CeedVectorRestoreArray(boundary_data->storage.ceed.vec, &boundary_data->storage.ceed.data));
    }
  } else {
    if (boundary_data->storage.updated) {
      PetscCall(VecRestoreArray(boundary_data->storage.petsc.vec, &boundary_data->storage.petsc.data));
    }
  }
  boundary_data->rdy->lock.boundary_data[boundary_data->boundary.index] = NULL;
  *boundary_data                                                        = (OperatorBoundaryData){0};
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GetOperatorSourceData(RDy rdy, OperatorSourceData *source_data) {
  PetscFunctionBegin;
  *source_data = (OperatorSourceData){0};
  PetscCheck(!rdy->lock.source_data, rdy->comm, PETSC_ERR_USER, "Could not acquire lock on source data -- another entity has access");
  rdy->lock.source_data = source_data;
  source_data->rdy      = rdy;
  PetscCall(VecGetBlockSize(rdy->u_global, &source_data->num_components));

  if (CeedEnabled(rdy)) {
    // NOTE: our SWE-specific source operator has only one sub operator
    CeedOperator *sub_ops;
    PetscCallCEED(CeedCompositeOperatorGetSubList(rdy->ceed.source_operator, &sub_ops));
    CeedOperator source_op = sub_ops[0];

    // fetch the relevant vector
    CeedOperatorField field;
    PetscCallCEED(CeedOperatorGetFieldByName(source_op, "swe_src", &field));  // FIXME: only valid for SWE
    PetscCallCEED(CeedOperatorFieldGetVector(field, &source_data->sources.ceed.vec));
  } else {
    source_data->sources.petsc.vec = rdy->petsc.sources;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// sets values of the source term for the given component
PetscErrorCode SetOperatorSourceValues(OperatorSourceData *source_data, PetscInt component, PetscReal *source_values) {
  PetscFunctionBegin;

  if (CeedEnabled(source_data->rdy)) {
    // if this is the first update, get access to the vector's data
    if (!source_data->sources.updated) {
      PetscCallCEED(CeedVectorGetArray(source_data->sources.ceed.vec, CEED_MEM_HOST, &source_data->sources.ceed.data));
      source_data->sources.updated = PETSC_TRUE;
    }

    // reshape for multicomponent access
    CeedScalar(*values)[source_data->num_components];
    *((CeedScalar **)&values) = source_data->sources.ceed.data;

    // set the values
    for (CeedInt i = 0; i < source_data->rdy->mesh.num_owned_cells; ++i) {
      values[i][component] = source_values[i];
    }
  } else {
    // if this is the first update, get access to the vector's data
    if (!source_data->sources.updated) {
      PetscCall(VecGetArray(source_data->sources.petsc.vec, &source_data->sources.petsc.data));
      source_data->sources.updated = PETSC_TRUE;
    }

    // set the values
    PetscReal *s = source_data->sources.petsc.data;
    for (PetscInt i = 0; i < source_data->rdy->mesh.num_owned_cells; ++i) {
      s[i * source_data->num_components + component] = source_values[i];
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RestoreOperatorSourceData(RDy rdy, OperatorSourceData *source_data) {
  PetscFunctionBegin;
  PetscCheck(rdy == source_data->rdy, rdy->comm, PETSC_ERR_USER, "Could not restore operator source data: wrong RDy");
  if (CeedEnabled(source_data->rdy)) {
    if (source_data->sources.updated) {
      PetscCallCEED(CeedVectorRestoreArray(source_data->sources.ceed.vec, &source_data->sources.ceed.data));
    }
    if (source_data->mannings.updated) {
      PetscCallCEED(CeedVectorRestoreArray(source_data->mannings.ceed.vec, &source_data->mannings.ceed.data));
    }
    if (source_data->flux_divergence.updated) {
      PetscCallCEED(CeedVectorRestoreArray(source_data->flux_divergence.ceed.vec, &source_data->flux_divergence.ceed.data));
    }
  } else {  // petsc
    if (source_data->sources.updated) {
      PetscCall(VecRestoreArray(source_data->sources.petsc.vec, &source_data->sources.petsc.data));
    }
    if (source_data->mannings.updated) {
      PetscCall(VecRestoreArray(source_data->mannings.petsc.vec, &source_data->mannings.petsc.data));
    }
    if (source_data->flux_divergence.updated) {
      PetscCall(VecRestoreArray(source_data->flux_divergence.petsc.vec, &source_data->flux_divergence.petsc.data));
    }
  }
  source_data->rdy->lock.source_data = NULL;
  *source_data                       = (OperatorSourceData){0};
  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
