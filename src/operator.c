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

// operator logging/profiling events
PetscLogEvent RDY_CeedOperatorApply_;

// defines the computational domain for the operator
static PetscErrorCode SetOperatorDomain(Operator *op, DM dm, RDyMesh *mesh) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCheck(mesh, comm, PETSC_ERR_USER, "Operator mesh must be non-NULL");

  op->dm   = dm;
  op->mesh = mesh;

  // create a global vector to store flux divergences
  PetscCall(DMCreateGlobalVector(op->dm, &op->flux_divergence));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// creates a sequential vector appropriate for our chosen architecture
static PetscErrorCode CreateSequentialVector(MPI_Comm comm, PetscInt block_size, PetscInt size, Vec *vec) {
  PetscFunctionBegin;

  VecType seq_vec_type;
  if (CeedEnabled()) {
    VecType ceed_vec_type = NULL;
    PetscCall(GetCeedVecType(&ceed_vec_type));
    if (!strcmp(VECCUDA, ceed_vec_type)) {
      seq_vec_type = VECSEQCUDA;
    } else if (!strcmp(VECKOKKOS, ceed_vec_type)) {
      seq_vec_type = VECSEQKOKKOS;
    } else if (!strcmp(VECHIP, ceed_vec_type)) {
      seq_vec_type = VECSEQHIP;
    } else {
      PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "Unsupported CEED vector type: %s", ceed_vec_type);
    }
  } else {
    seq_vec_type = VECSEQ;
  }
  PetscCall(VecCreate(PETSC_COMM_SELF, vec));
  PetscCall(VecSetSizes(*vec, size, size));
  PetscCall(VecSetType(*vec, seq_vec_type));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// defines the distinct regions of owned cells for the operator
static PetscErrorCode SetOperatorRegions(Operator *op, PetscInt num_regions, RDyRegion *regions) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCheck(num_regions > 0, comm, PETSC_ERR_USER, "Number of operator regions must be positive");
  PetscCheck(regions, comm, PETSC_ERR_USER, "Operator region array must be non-NULL");

  PetscCall(PetscCalloc1(num_regions, &op->regions));
  memcpy(op->regions, regions, sizeof(RDyRegion) * num_regions);

  // allocate sequential vectors for each region
  if (!CeedEnabled()) {
    PetscCall(PetscCalloc1(num_regions, &op->petsc.sources));
    for (PetscInt r = 0; r < num_regions; ++r) {
      PetscCall(CreateSequentialVector(comm, op->num_components, regions[r].num_owned_cells, &op->petsc.sources[r]));
    }

    PetscCall(PetscCalloc1(OPERATOR_NUM_MATERIAL_PROPERTIES, &op->petsc.material_properties));
    for (PetscInt p = 0; p < OPERATOR_NUM_MATERIAL_PROPERTIES; ++p) {
      PetscCall(PetscCalloc1(num_regions, &op->petsc.material_properties[p]));
      for (PetscInt r = 0; r < num_regions; ++r) {
        PetscCall(CreateSequentialVector(comm, op->num_components, regions[r].num_owned_cells, &op->petsc.material_properties[p][r]));
      }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// defines the distinct boundaries (edges) for the operator
static PetscErrorCode SetOperatorBoundaries(Operator *op, PetscInt num_boundaries, RDyBoundary *boundaries) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCheck(num_boundaries > 0, comm, PETSC_ERR_USER, "Number of operator boundaries must be positive");
  PetscCheck(boundaries, comm, PETSC_ERR_USER, "Operator boundary array must be non-NULL");

  PetscCall(PetscCalloc1(num_boundaries, &op->boundaries));
  memcpy(op->boundaries, boundaries, sizeof(RDyBoundary) * num_boundaries);

  if (!CeedEnabled()) {
    PetscCall(PetscCalloc1(num_boundaries, &op->petsc.boundary_values));
    PetscCall(PetscCalloc1(num_boundaries, &op->petsc.boundary_fluxes));

    for (PetscInt b = 0; b < num_boundaries; ++b) {
      PetscInt num_edges = boundaries[b].num_edges;
      PetscCall(CreateSequentialVector(comm, op->num_components, num_edges, &op->petsc.boundary_values[b]));
      PetscCall(CreateSequentialVector(comm, op->num_components, num_edges, &op->petsc.boundary_fluxes[b]));
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrepareCeedOperator(Operator *op) {
  PetscFunctionBegin;

  Ceed ceed = CeedContext();
  PetscCallCEED(CeedCompositeOperatorCreate(ceed, &op->ceed.composite));

  PetscReal tiny_h = op->physics_config.flow.tiny_h;

  // set up suboperators for the shallow water equations

  // suboperator 0: fluxes between interior cells
  CeedOperator interior_flux_op;
  PetscCall(CreateSWEInteriorFluxOperator(ceed, op->mesh, tiny_h, &interior_flux_op));
  PetscCallCEED(CeedCompositeOperatorAddSub(op->ceed.composite, interior_flux_op));
  PetscCallCEED(CeedOperatorDestroy(&interior_flux_op));

  // suboperators 1 to num_boundaries: fluxes on boundary edges
  for (CeedInt b = 0; b < op->num_boundaries; ++b) {
    CeedOperator boundary_op;
    RDyBoundary  boundary           = op->boundaries[b];
    RDyCondition boundary_condition = op->boundary_conditions[b];
    PetscCall(CreateSWEBoundaryFluxOperator(ceed, op->mesh, op->boundary, tiny_h, &boundary_flux_op));
    PetscCallCEED(CeedCompositeOperatorAddSub(op->composite, boundary_flux_op));
    PetscCallCEED(CeedOperatorDestroy(&boundary_flux_op));
  }

  // suboperators num_boundaries + 1 to num_boundaries + num_regions + 1: external sources
  for (CeedInt r = 0; r < op->num_regions; ++r) {
    CeedOperator source_op;
    RDyRegion region = op->regions[r];
    PetscCall(CreateSWEExternalSourceOperator(ceed, op->mesh, tiny_h, &source_op));
    PetscCallCEED(CeedCompositeOperatorAddSub(op->composite, source_op));
    PetscCallCEED(CeedOperatorDestroy(&source_op));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PreparePetscOperator(Operator *op) {
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// performs all work necessary to make the operator ready for use
static PetscErrorCode PrepareOperator(Operator *op) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));

  if (CeedEnabled()) {
    PetscCall(PrepareCeedOperator(op));
  } else {
    PetscCall(PreparePetscOperator(op));
  }

  // initialize diagnostics
  op->courant_number_diags.max_courant_num = 0.0;
  op->courant_number_diags.global_edge_id  = -1;
  op->courant_number_diags.global_cell_id  = -1;
  op->courant_number_diags.is_set          = PETSC_FALSE;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Creates an operator representing the system of equations described in the
// given configuration.
PetscErrorCode CreateOperator(RDyPhysicsSection physics_config, DM domain_dm, RDyMesh *domain_mesh, PetscInt num_regions, RDyRegion *regions,
                              PetscInt num_boundaries, RDyBoundary *boundaries, Operator **operator) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)domain_dm, &comm));

  // check our arguments
  PetscCheck(domain_mesh, comm, PETSC_ERR_USER, "Cannot create an operator with no mesh");
  PetscCheck(num_regions > 0, comm, PETSC_ERR_USER, "Cannot create an operator with no regions");
  PetscCheck(num_boundaries > 0, comm, PETSC_ERR_USER, "Cannot create an operator with no boundaries");

  static PetscBool first_time = PETSC_TRUE;
  if (first_time) {
    // register a logging event for applying our CEED operator
    PetscCall(PetscLogEventRegister("CeedOperatorApp", RDY_CLASSID, &RDY_CeedOperatorApply_));
    first_time = PETSC_FALSE;
  }

  PetscCall(PetscCalloc1(1, operator));
  (*operator)->physics_config = physics_config;

  PetscCall(SetOperatorDomain(*operator, domain_dm, domain_mesh));
  PetscCall(SetOperatorBoundaries(*operator, num_boundaries, boundaries));
  PetscCall(SetOperatorRegions(*operator, num_regions, regions));
  PetscCall(ReadyOperator(*operator));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Free all resources devoted to the operator.
PetscErrorCode DestroyOperator(Operator **op) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)(*op)->dm, &comm));

  PetscBool ceed_enabled = CeedEnabled();

  if ((*op)->petsc.context) {
    PetscCall(DestroyPetscSWEFlux((*op)->petsc.context, ceed_enabled, (*op)->num_boundaries));
  }

  if (ceed_enabled) {
    PetscCallCEED(CeedOperatorDestroy(&((*op)->ceed.composite)));
    PetscCallCEED(CeedVectorDestroy(&((*op)->ceed.u_local)));
    PetscCallCEED(CeedVectorDestroy(&((*op)->ceed.rhs)));
    PetscCallCEED(CeedVectorDestroy(&((*op)->ceed.sources)));
  } else {
    for (PetscInt b = 0; b < (*op)->num_boundaries; ++b) {
      PetscCall(VecDestroy(&(*op)->petsc.boundary_values[b]));
      PetscCall(VecDestroy(&(*op)->petsc.boundary_fluxes[b]));
    }
    PetscFree((*op)->petsc.boundary_values);
    PetscFree((*op)->petsc.boundary_fluxes);
    for (PetscInt r = 0; r < (*op)->num_regions; ++r) {
      PetscCall(VecDestroy(&(*op)->petsc.sources[r]));
    }
    PetscFree((*op)->petsc.sources);
    for (PetscInt p = 0; p < 1; ++p) {
      for (PetscInt r = 0; r < (*op)->num_regions; ++r) {
        PetscCall(VecDestroy(&(*op)->petsc.material_properties[p][r]));
      }
      PetscFree((*op)->petsc.material_properties[p]);
    }
    PetscFree((*op)->petsc.material_properties);
  }

  PetscFree(*op);
  *op = NULL;

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ApplyCeedOperator(Operator *op, PetscReal dt, Vec u, Vec dudt) {
  PetscFunctionBegin;

  // update the timestep for the ceed operators if necessary
  if (op->ceed.dt != dt) {
    op->ceed.dt = dt;
    CeedContextFieldLabel label;
    PetscCallCEED(CeedOperatorGetContextFieldLabel(op->ceed.composite, "time step", &label));
    PetscCallCEED(CeedOperatorSetContextDouble(op->ceed.composite, label, &op->ceed.dt));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ApplyPetscOperator(Operator *op, PetscReal dt, Vec u, Vec dudt) {
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ApplyOperator(Operator *op, PetscReal dt, Vec u, Vec dudt) {
  PetscFunctionBegin;

  if (CeedEnabled()) {
    PetscCall(ApplyCeedOperator(op, dt, u, dudt));
  } else {
    PetscCall(ApplyPetscOperator(op, dt, u, dudt));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GetOperatorCourantNumberDiagnostics(Operator *op, CourantNumberDiagnostics *diags) {
  PetscFunctionBegin;

  // reduce the local courant number diagnostics if needed
  if (!op->courant_number_diags.is_set) {
    MPI_Comm comm;
    PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
    MPI_Allreduce(MPI_IN_PLACE, &op->courant_number_diags, 1, MPI_COURANT_NUMBER_DIAGNOSTICS, MPI_MAX_COURANT_NUMBER, comm);
    op->courant_number_diags.is_set = PETSC_TRUE;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

//----------------------------------------
// Boundary (Edge-Centered) Operator Data
//----------------------------------------

static PetscErrorCode CheckOperatorBoundary(Operator *op, RDyBoundary boundary, MPI_Comm comm) {
  PetscFunctionBegin;
  PetscCheck(boundary.index >= 0 && boundary.index < op->num_boundaries, comm, PETSC_ERR_USER,
             "Invalid boundary for operator (index: %" PetscInt_FMT ")", boundary.index);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GetCeedOperatorBoundaryData(Operator *op, RDyBoundary boundary, const char *field_name, OperatorData *boundary_data) {
  PetscFunctionBegin;

  // get the relevant boundary sub-operator
  CeedOperator *sub_ops;
  PetscCallCEED(CeedCompositeOperatorGetSubList(op->ceed.composite, &sub_ops));
  CeedOperator sub_op = sub_ops[1 + boundary.index];

  // fetch the relevant vector
  CeedOperatorField field;
  PetscCallCEED(CeedOperatorGetFieldByName(sub_op, field_name, &field));
  CeedVector vec;
  PetscCallCEED(CeedOperatorFieldGetVector(field, &vec));

  // copy operator data into place
  PetscCallCEED(CeedVectorGetArray(vec, CEED_MEM_HOST, (CeedScalar **)&boundary_data->array_pointer));
  PetscInt num_components = op->num_components;
  CeedScalar(*values)[num_components];
  *((CeedScalar **)&values) = (CeedScalar *)boundary_data->array_pointer;
  for (PetscInt c = 0; c < num_components; ++c) {
    boundary_data->values[c] = values[c];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RestoreCeedOperatorBoundaryData(Operator *op, RDyBoundary boundary, const char *field_name, OperatorData *boundary_data) {
  PetscFunctionBegin;

  // get the relevant boundary sub-operator
  CeedOperator *sub_ops;
  PetscCallCEED(CeedCompositeOperatorGetSubList(op->ceed.composite, &sub_ops));
  CeedOperator sub_op = sub_ops[1 + boundary.index];

  // fetch the relevant vector
  CeedOperatorField field;
  PetscCallCEED(CeedOperatorGetFieldByName(sub_op, field_name, &field));
  CeedVector vec;
  PetscCallCEED(CeedOperatorFieldGetVector(field, &vec));

  // release the array
  PetscCallCEED(CeedVectorRestoreArray(vec, (CeedScalar **)&boundary_data->array_pointer));
  PetscFree(boundary_data->values);
  *boundary_data = (OperatorData){0};

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GetPetscOperatorBoundaryData(Operator *op, RDyBoundary boundary, Vec vec, OperatorData *boundary_data) {
  PetscFunctionBegin;

  PetscReal *data;
  PetscCall(VecGetArray(vec, &data));
  for (PetscInt c = 0; c < op->num_components; ++c) {
    PetscCall(PetscCalloc1(boundary.num_edges, &boundary_data->values[c]));
    for (PetscInt e = 0; e < boundary.num_edges; ++e) {
      boundary_data->values[c][e] = data[op->num_components * e + c];
    }
  }
  boundary_data->array_pointer = data;

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RestorePetscOperatorBoundaryData(Operator *op, RDyBoundary boundary, Vec vec, OperatorData *boundary_data) {
  PetscFunctionBegin;

  PetscReal *data = boundary_data->array_pointer;
  for (PetscInt c = 0; c < op->num_components; ++c) {
    for (PetscInt e = 0; e < boundary.num_edges; ++e) {
      data[op->num_components * e + c] = boundary_data->values[c][e];
    }
  }
  PetscCall(VecRestoreArray(vec, &data));
  *boundary_data = (OperatorData){0};

  PetscFunctionReturn(PETSC_SUCCESS);
}

// acquires exclusive access to boundary values for the operator
PetscErrorCode GetOperatorBoundaryValues(Operator *op, RDyBoundary boundary, OperatorData *boundary_values) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCall(CheckOperatorBoundary(op, boundary, comm));

  boundary_values->num_components = op->num_components;
  PetscCall(PetscCalloc1(op->num_components, &boundary_values->values));
  if (CeedEnabled()) {
    PetscCall(GetCeedOperatorBoundaryData(op, boundary, "q_dirichlet", boundary_values));
  } else {  // petsc
    PetscCall(GetPetscOperatorBoundaryData(op, boundary, op->petsc.boundary_values[boundary.index], boundary_values));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RestoreOperatorBoundaryValues(Operator *op, RDyBoundary boundary, OperatorData *boundary_values) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCall(CheckOperatorBoundary(op, boundary, comm));

  if (CeedEnabled()) {
    PetscCallCEED(RestoreCeedOperatorBoundaryData(op, boundary, "q_dirichlet", boundary_values));
  } else {
    PetscCallCEED(RestorePetscOperatorBoundaryData(op, boundary, op->petsc.boundary_values[boundary.index], boundary_values));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// acquires exclusive access to boundary fluxes for the operator
PetscErrorCode GetOperatorBoundaryFluxes(Operator *op, RDyBoundary boundary, OperatorData *boundary_fluxes) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCall(CheckOperatorBoundary(op, boundary, comm));

  boundary_fluxes->num_components = op->num_components;
  PetscCall(PetscCalloc1(op->num_components, &boundary_fluxes->values));
  if (CeedEnabled()) {
    PetscCall(GetCeedOperatorBoundaryData(op, boundary, "flux", boundary_fluxes));
  } else {  // petsc
    PetscCall(GetPetscOperatorBoundaryData(op, boundary, op->petsc.boundary_fluxes[boundary.index], boundary_fluxes));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RestoreOperatorBoundaryFluxes(Operator *op, RDyBoundary boundary, OperatorData *boundary_fluxes) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCall(CheckOperatorBoundary(op, boundary, comm));

  if (CeedEnabled()) {
    PetscCallCEED(RestoreCeedOperatorBoundaryData(op, boundary, "flux", boundary_fluxes));
  } else {
    PetscCallCEED(RestorePetscOperatorBoundaryData(op, boundary, op->petsc.boundary_fluxes[boundary.index], boundary_fluxes));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

//----------------------------------------
// Regional (Cell-Centered) Operator Data
//----------------------------------------

static PetscErrorCode CheckOperatorRegion(Operator *op, RDyRegion region, MPI_Comm comm) {
  PetscFunctionBegin;

  PetscCheck(region.index >= 0 && region.index < op->num_regions, comm, PETSC_ERR_USER, "Invalid region for source data (index: %" PetscInt_FMT ")",
             region.index);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GetCeedOperatorRegionData(Operator *op, RDyRegion region, const char *field_name, OperatorData *region_data) {
  PetscFunctionBegin;

  CeedOperator *sub_ops;
  PetscCallCEED(CeedCompositeOperatorGetSubList(op->ceed.composite, &sub_ops));
  CeedOperator source_op = sub_ops[1 + op->num_boundaries + region.index];

  // fetch the relevant vector
  CeedOperatorField field;
  PetscCallCEED(CeedOperatorGetFieldByName(source_op, field_name, &field));
  CeedVector vec;
  PetscCallCEED(CeedOperatorFieldGetVector(field, &vec));

  // copy operator data into place
  PetscCallCEED(CeedVectorGetArray(vec, CEED_MEM_HOST, (CeedScalar **)&region_data->array_pointer));
  PetscInt num_components = op->num_components;
  CeedScalar(*values)[num_components];
  *((CeedScalar **)&values) = (CeedScalar *)region_data->array_pointer;
  for (PetscInt c = 0; c < num_components; ++c) {
    region_data->values[c] = values[c];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RestoreCeedOperatorRegionData(Operator *op, RDyRegion region, const char *field_name, OperatorData *region_data) {
  PetscFunctionBegin;

  CeedOperator *sub_ops;
  PetscCallCEED(CeedCompositeOperatorGetSubList(op->ceed.composite, &sub_ops));
  CeedOperator source_op = sub_ops[1 + op->num_boundaries + region.index];

  // fetch the relevant vector
  CeedOperatorField field;
  PetscCallCEED(CeedOperatorGetFieldByName(source_op, field_name, &field));
  CeedVector vec;
  PetscCallCEED(CeedOperatorFieldGetVector(field, &vec));

  // release the array
  PetscCallCEED(CeedVectorRestoreArray(vec, (CeedScalar **)&region_data->array_pointer));
  PetscFree(region_data->values);
  *region_data = (OperatorData){0};

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GetPetscOperatorRegionData(Operator *op, RDyRegion region, Vec vec, OperatorData *region_data) {
  PetscFunctionBegin;

  PetscReal *data;
  PetscCall(VecGetArray(vec, &data));
  for (PetscInt c = 0; c < op->num_components; ++c) {
    PetscCall(PetscCalloc1(region.num_owned_cells, &region_data->values[c]));
    for (PetscInt ce = 0; ce < region.num_owned_cells; ++ce) {
      region_data->values[c][ce] = data[op->num_components * ce + c];
    }
  }
  region_data->array_pointer = data;

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RestorePetscOperatorRegionData(Operator *op, RDyRegion region, Vec vec, OperatorData *region_data) {
  PetscFunctionBegin;

  PetscReal *data = region_data->array_pointer;
  for (PetscInt c = 0; c < op->num_components; ++c) {
    for (PetscInt ce = 0; ce < region.num_owned_cells; ++ce) {
      data[op->num_components * ce + c] = region_data->values[c][ce];
    }
  }
  PetscCall(VecRestoreArray(vec, &data));
  *region_data = (OperatorData){0};

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GetOperatorExternalSource(Operator *op, RDyRegion region, OperatorData *source_data) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCall(CheckOperatorRegion(op, region, comm));

  source_data->num_components = op->num_components;
  PetscCall(PetscCalloc1(op->num_components, &source_data->values));
  if (CeedEnabled()) {
    PetscCall(GetCeedOperatorRegionData(op, region, "swe_src", source_data));
  } else {  // petsc
    PetscCall(GetPetscOperatorRegionData(op, region, op->petsc.sources[region.index], source_data));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RestoreOperatorExternalSource(Operator *op, RDyRegion region, OperatorData *source_data) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCall(CheckOperatorRegion(op, region, comm));

  if (CeedEnabled()) {
    PetscCallCEED(RestoreCeedOperatorRegionData(op, region, "swe_src", source_data));
  } else {
    PetscCallCEED(RestorePetscOperatorRegionData(op, region, op->petsc.sources[region.index], source_data));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------
// Regional Material Property Operator Data
//------------------------------------------

const char *material_property_names[] = {
    "manning",
};

PetscErrorCode GetOperatorMaterialProperty(Operator *op, RDyRegion region, OperatorMaterialPropertyId property_id, OperatorData *property) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCall(CheckOperatorRegion(op, region, comm));

  switch (property_id) {
    case OPERATOR_MANNINGS:
      if (CeedEnabled()) {
        PetscCall(GetCeedOperatorRegionData(op, region, material_property_names[property_id], property));
      } else {
        PetscCall(GetPetscOperatorRegionData(op, region, op->petsc.material_properties[region.index][property_id], property));
      }
    default:
      PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "Invalid material property ID: %u", property_id);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RestoreOperatorMaterialProperty(Operator *op, RDyRegion region, OperatorMaterialPropertyId property_id, OperatorData *property) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCall(CheckOperatorRegion(op, region, comm));

  switch (property_id) {
    case OPERATOR_MANNINGS:
      if (CeedEnabled()) {
        PetscCall(RestoreCeedOperatorRegionData(op, region, material_property_names[property_id], property));
      } else {
        PetscCall(RestorePetscOperatorRegionData(op, region, op->petsc.material_properties[region.index][property_id], property));
      }
    default:
      PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "Invalid material property ID: %u", property_id);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
