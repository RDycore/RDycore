#include <ceed/ceed.h>
#include <petscdmceed.h>
#include <private/rdyoperatorimpl.h>
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

  PetscBool ceed_enabled = CeedEnabled();

  if (rdy->petsc.context) {
    PetscCall(DestroyPetscSWEFlux(rdy->petsc.context, ceed_enabled, rdy->num_boundaries));
  }

  if (ceed_enabled) {
    PetscCallCEED(CeedOperatorDestroy(&rdy->ceed.flux_operator));
    PetscCallCEED(CeedOperatorDestroy(&rdy->ceed.source_operator));
    PetscCallCEED(CeedVectorDestroy(&rdy->ceed.u_local));
    PetscCallCEED(CeedVectorDestroy(&rdy->ceed.rhs));
    PetscCallCEED(CeedVectorDestroy(&rdy->ceed.sources));
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

  if (CeedEnabled()) {
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

  if (CeedEnabled()) {
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

PetscErrorCode RestoreOperatorBoundaryData(RDy rdy, RDyBoundary boundary, OperatorBoundaryData *boundary_data) {
  PetscFunctionBegin;
  PetscCheck(rdy == boundary_data->rdy, rdy->comm, PETSC_ERR_USER, "Could not restore operator boundary data: wrong RDy");
  PetscCheck(boundary.index == boundary_data->boundary.index, rdy->comm, PETSC_ERR_USER, "Could not restore operator boundary data: wrong boundary");
  if (CeedEnabled()) {
    if (boundary_data->storage.updated) {
      PetscCallCEED(CeedVectorRestoreArray(boundary_data->storage.ceed.vec, &boundary_data->storage.ceed.data));
    }
  } else {
    if (boundary_data->storage.updated) {
      PetscCall(VecRestoreArray(boundary_data->storage.petsc.vec, &boundary_data->storage.petsc.data));
    }
  }
  boundary_data->rdy->lock.boundary_data[boundary.index] = NULL;
  *boundary_data                                         = (OperatorBoundaryData){0};
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GetOperatorSourceData(RDy rdy, RDyRegion region, OperatorSourceData *source_data) {
  PetscFunctionBegin;
  *source_data = (OperatorSourceData){0};
  PetscCheck(!rdy->lock.source_data, rdy->comm, PETSC_ERR_USER, "Could not acquire lock on source data -- another entity has access");
  PetscCheck(region.index >= 0 && region.index < rdy->num_regions, rdy->comm, PETSC_ERR_USER,
             "Invalid region for source data (index: %" PetscInt_FMT ")", region.index);
  rdy->lock.source_data = source_data;
  source_data->rdy      = rdy;
  source_data->region   = region;
  PetscCall(VecGetBlockSize(rdy->u_global, &source_data->num_components));

  if (CeedEnabled()) {
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

  // if this is the first update, get access to the vector's data
  // (only the source data that obtained the lock actually gains access to
  //  the array)
  OperatorSourceData *source_data_with_lock = source_data->rdy->lock.source_data;
  if (!source_data->sources.updated) {
    if (source_data == source_data_with_lock) {
      if (CeedEnabled()) {
        PetscCallCEED(CeedVectorGetArray(source_data->sources.ceed.vec, CEED_MEM_HOST, &source_data->sources.ceed.data));
      } else {
        PetscCall(VecGetArray(source_data->sources.petsc.vec, &source_data->sources.petsc.data));
      }
    } else {
      source_data->sources.ceed.data = source_data_with_lock->sources.ceed.data;
    }
    source_data->sources.updated = PETSC_TRUE;
  }

  RDyMesh  *mesh  = &source_data->rdy->mesh;
  RDyCells *cells = &mesh->cells;

  // fetch values -- the underlying vector is the set of all owned cells in
  // the computational domain, so we have to sift through indices
  if (CeedEnabled()) {
    // reshape for multicomponent access
    CeedScalar(*values)[source_data->num_components];
    *((CeedScalar **)&values) = source_data->sources.ceed.data;

    for (PetscInt c = 0; c < source_data->region.num_cells; ++c) {
      PetscInt cell_id = source_data->region.cell_ids[c];
      if (cells->is_local[cell_id]) {
        PetscInt owned_cell_id           = cells->local_to_owned[cell_id];
        values[owned_cell_id][component] = source_values[c];
      }
    }
  } else {
    PetscReal *s = source_data->sources.petsc.data;
    for (PetscInt c = 0; c < source_data->region.num_cells; ++c) {
      PetscInt cell_id = source_data->region.cell_ids[c];
      if (cells->is_local[cell_id]) {
        PetscInt owned_cell_id                                     = cells->local_to_owned[cell_id];
        s[owned_cell_id * source_data->num_components + component] = source_values[c];
      }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// sets values of the source term for the given component
PetscErrorCode GetOperatorSourceValues(OperatorSourceData *source_data, PetscInt component, PetscReal *source_values) {
  PetscFunctionBegin;

  // if this is the first update, get access to the vector's data
  // (only the source data that obtained the lock actually gains access to
  //  the array)
  OperatorSourceData *source_data_with_lock = source_data->rdy->lock.source_data;
  if (!source_data->sources.updated) {
    if (source_data == source_data_with_lock) {
      if (CeedEnabled()) {
        PetscCallCEED(CeedVectorGetArray(source_data->sources.ceed.vec, CEED_MEM_HOST, &source_data->sources.ceed.data));
      } else {
        PetscCall(VecGetArray(source_data->sources.petsc.vec, &source_data->sources.petsc.data));
      }
    } else {
      source_data->sources.ceed.data = source_data_with_lock->sources.ceed.data;
    }
    source_data->sources.updated = PETSC_TRUE;
  }

  RDyMesh  *mesh  = &source_data->rdy->mesh;
  RDyCells *cells = &mesh->cells;

  // fetch values -- the underlying vector is the set of all owned cells in
  // the computational domain, so we have to sift through indices
  if (CeedEnabled()) {
    // reshape for multicomponent access
    CeedScalar(*values)[source_data->num_components];
    *((CeedScalar **)&values) = source_data->sources.ceed.data;

    for (PetscInt c = 0; c < source_data->region.num_cells; ++c) {
      PetscInt cell_id = source_data->region.cell_ids[c];
      if (cells->is_local[cell_id]) {
        PetscInt owned_cell_id = cells->local_to_owned[cell_id];
        source_values[c]       = values[owned_cell_id][component];
      }
    }
  } else {
    PetscReal *s = source_data->sources.petsc.data;

    for (PetscInt c = 0; c < source_data->region.num_cells; ++c) {
      PetscInt cell_id = source_data->region.cell_ids[c];
      if (cells->is_local[cell_id]) {
        PetscInt owned_cell_id = cells->local_to_owned[cell_id];
        source_values[c]       = s[owned_cell_id * source_data->num_components + component];
      }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RestoreOperatorSourceData(RDy rdy, RDyRegion region, OperatorSourceData *source_data) {
  PetscFunctionBegin;
  PetscCheck(rdy == source_data->rdy, rdy->comm, PETSC_ERR_USER, "Could not restore operator source data: wrong RDy");
  PetscCheck(region.index == source_data->region.index, rdy->comm, PETSC_ERR_USER, "Could not restore operator source data: wrong region");

  // only the source data that obtained the lock actually gains access to
  // the array
  OperatorSourceData *source_data_with_lock = source_data->rdy->lock.source_data;
  if (source_data->sources.updated && source_data == source_data_with_lock) {
    if (CeedEnabled()) {
      PetscCallCEED(CeedVectorRestoreArray(source_data->sources.ceed.vec, &source_data->sources.ceed.data));
    } else {  // petsc
      PetscCall(VecRestoreArray(source_data->sources.petsc.vec, &source_data->sources.petsc.data));
    }
  }
  source_data->rdy->lock.source_data = NULL;
  *source_data                       = (OperatorSourceData){0};
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GetOperatorMaterialData(RDy rdy, OperatorMaterialData *material_data) {
  PetscFunctionBegin;
  *material_data = (OperatorMaterialData){0};
  PetscCheck(!rdy->lock.material_data, rdy->comm, PETSC_ERR_USER, "Could not acquire lock on material data -- another entity has access");
  rdy->lock.material_data = material_data;
  material_data->rdy      = rdy;

  if (CeedEnabled()) {
    // NOTE: our SWE-specific source operator has only one sub operator
    CeedOperator *sub_ops;
    PetscCallCEED(CeedCompositeOperatorGetSubList(rdy->ceed.source_operator, &sub_ops));
    CeedOperator source_op = sub_ops[0];

    // fetch the relevant material property vectors
    CeedOperatorField field;
    PetscCallCEED(CeedOperatorGetFieldByName(source_op, "mannings_s", &field));  // FIXME: only valid for SWE
    PetscCallCEED(CeedOperatorFieldGetVector(field, &material_data->mannings.ceed.vec));
  } else {
    material_data->mannings.petsc.vec = rdy->petsc.sources;  // FIXME: incorrect!
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// sets values of the given operator material property
PetscErrorCode SetOperatorMaterialValues(OperatorMaterialData *material_data, OperatorMaterialDataIndex index, PetscReal *material_values) {
  PetscFunctionBegin;

  // pick the appropriate data vector
  OperatorVectorData vector_data;
  switch (index) {
    case OPERATOR_MANNINGS:
      vector_data = material_data->mannings;
      break;
  }

  if (CeedEnabled()) {
    // if this is the first update, get access to the vector's data
    if (!vector_data.updated) {
      PetscCallCEED(CeedVectorGetArray(vector_data.ceed.vec, CEED_MEM_HOST, &vector_data.ceed.data));
      vector_data.updated = PETSC_TRUE;
    }

    CeedScalar *values = vector_data.ceed.data;

    // set the values
    for (CeedInt i = 0; i < material_data->rdy->mesh.num_owned_cells; ++i) {
      values[i] = material_values[i];
    }
  } else {
    // if this is the first update, get access to the vector's data
    if (!vector_data.updated) {
      PetscCall(VecGetArray(vector_data.petsc.vec, &vector_data.petsc.data));
      vector_data.updated = PETSC_TRUE;
    }

    // set the values
    PetscReal *m = vector_data.petsc.data;
    for (PetscInt i = 0; i < material_data->rdy->mesh.num_owned_cells; ++i) {
      m[i] = material_values[i];
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RestoreOperatorMaterialData(RDy rdy, OperatorMaterialData *material_data) {
  PetscFunctionBegin;
  PetscCheck(rdy == material_data->rdy, rdy->comm, PETSC_ERR_USER, "Could not restore operator material data: wrong RDy");
  if (CeedEnabled()) {
    if (material_data->mannings.updated) {
      PetscCallCEED(CeedVectorRestoreArray(material_data->mannings.ceed.vec, &material_data->mannings.ceed.data));
    }
  } else {  // petsc
    if (material_data->mannings.updated) {
      PetscCall(VecRestoreArray(material_data->mannings.petsc.vec, &material_data->mannings.petsc.data));
    }
  }
  material_data->rdy->lock.material_data = NULL;
  *material_data                         = (OperatorMaterialData){0};
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GetOperatorFluxDivergenceData(RDy rdy, OperatorFluxDivergenceData *flux_div_data) {
  PetscFunctionBegin;
  *flux_div_data = (OperatorFluxDivergenceData){0};
  PetscCheck(!rdy->lock.flux_div_data, rdy->comm, PETSC_ERR_USER, "Could not acquire lock on flux divergence data -- another entity has access");
  rdy->lock.flux_div_data = flux_div_data;
  flux_div_data->rdy      = rdy;
  PetscCall(VecGetBlockSize(rdy->u_global, &flux_div_data->num_components));

  if (CeedEnabled()) {
    // NOTE: our SWE-specific source operator has only one sub operator
    CeedOperator *sub_ops;
    PetscCallCEED(CeedCompositeOperatorGetSubList(rdy->ceed.source_operator, &sub_ops));
    CeedOperator source_op = sub_ops[0];

    // fetch the relevant vector
    CeedOperatorField field;
    PetscCallCEED(CeedOperatorGetFieldByName(source_op, "riemannf", &field));  // FIXME: only valid for SWE
    PetscCallCEED(CeedOperatorFieldGetVector(field, &flux_div_data->storage.ceed.vec));
  } else {
    flux_div_data->storage.petsc.vec = rdy->petsc.sources;  // FIXME: incorrect!
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// sets values of the source term for the given component
PetscErrorCode SetOperatorFluxDivergenceValues(OperatorFluxDivergenceData *flux_div_data, PetscInt component, PetscReal *flux_div_values) {
  PetscFunctionBegin;

  if (CeedEnabled()) {
    // if this is the first update, get access to the vector's data
    if (!flux_div_data->storage.updated) {
      PetscCallCEED(CeedVectorGetArray(flux_div_data->storage.ceed.vec, CEED_MEM_HOST, &flux_div_data->storage.ceed.data));
      flux_div_data->storage.updated = PETSC_TRUE;
    }

    // reshape for multicomponent access
    CeedScalar(*values)[flux_div_data->num_components];
    *((CeedScalar **)&values) = flux_div_data->storage.ceed.data;

    // set the values
    for (CeedInt i = 0; i < flux_div_data->rdy->mesh.num_owned_cells; ++i) {
      values[i][component] = flux_div_values[i];
    }
  } else {
    // if this is the first update, get access to the vector's data
    if (!flux_div_data->storage.updated) {
      PetscCall(VecGetArray(flux_div_data->storage.petsc.vec, &flux_div_data->storage.petsc.data));
      flux_div_data->storage.updated = PETSC_TRUE;
    }

    // set the values
    PetscReal *div_f = flux_div_data->storage.petsc.data;
    for (PetscInt i = 0; i < flux_div_data->rdy->mesh.num_owned_cells; ++i) {
      div_f[i * flux_div_data->num_components + component] = flux_div_values[i];
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RestoreOperatorFluxDivergenceData(RDy rdy, OperatorFluxDivergenceData *flux_div_data) {
  PetscFunctionBegin;
  PetscCheck(rdy == flux_div_data->rdy, rdy->comm, PETSC_ERR_USER, "Could not restore operator flux divergence data: wrong RDy");
  if (CeedEnabled()) {
    if (flux_div_data->storage.updated) {
      PetscCallCEED(CeedVectorRestoreArray(flux_div_data->storage.ceed.vec, &flux_div_data->storage.ceed.data));
    }
  } else {  // petsc
    if (flux_div_data->storage.updated) {
      PetscCall(VecRestoreArray(flux_div_data->storage.petsc.vec, &flux_div_data->storage.petsc.data));
    }
  }
  flux_div_data->rdy->lock.flux_div_data = NULL;
  *flux_div_data                         = (OperatorFluxDivergenceData){0};
  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
