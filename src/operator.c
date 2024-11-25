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
static PetscBool    operator_initialized_ = PETSC_FALSE;
static PetscClassId RDY_CLASSID_;
PetscLogEvent       RDY_CeedOperatorApply_;

// initializes operators for physics specified in the input configuration
static PetscErrorCode InitOperators(RDy rdy) {
  PetscFunctionBegin;

  // register a logging event for applying our CEED operator
  PetscCall(PetscClassIdRegister("RDycore", &RDY_CLASSID));
  PetscCall(PetscLogEventRegister("CeedOperatorApp", RDY_CLASSID, &RDY_CeedOperatorApply));

  // just pass the call along for now
  InitSWE(rdy);

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Creates an operator representing the system of equations described in the
// given configuration.
PetscErrorCode CreateOperator(RDyPhysicsSection physics_config, Operator **operator) {
  PetscFunctionBegin;

  PetscCall(PetscCalloc1(1, operator));
  (*operator)->physics_config = physics_config;

  // set up fields for the shallow water equations
  // FIXME: generalize
  (*operator)->num_components = 3;
  PetscCall(PetscCalloc1((*operator)->num_components, (*operator)->field_names));
  for (PetscInt c = 0; c < (*operator)->num_components; ++c) {
    PetscCall(PetscCalloc1(MAX_NAME_LEN + 1, &(*operator)->field_names[c]));
  }
  strcpy((*operator)->field_names[0], "Height");
  strcpy((*operator)->field_names[0], "MomentumX");
  strcpy((*operator)->field_names[0], "MomentumY");

  // initialize diagnostics
  (*operator)->local_courant_diags.max_courant_num = 0.0;
  (*operator)->local_courant_diags.global_edge_id  = -1;
  (*operator)->local_courant_diags.global_cell_id  = -1;
  (*operator)->local_courant_diags.is_set          = PETSC_FALSE;

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
    for (PetscInt r = 0; r < (*op)->num_regions; ++r) {
      PetscCall(VecDestroy(&(*op)->petsc.sources[r]));
    }
    for (PetscInt p = 0; p < 1; ++p) {
      for (PetscInt r = 0; r < (*op)->num_regions; ++r) {
        PetscCall(VecDestroy(&(*op)->petsc.material_properties[p][r]));
      }
    }
  }

  PetscFree(*op);
  *op = NULL;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// This function creates a Section from information in the given operator
/// appropriate for use by the given DM. It does not configure the DM to use
/// the section, however.
PetscErrorCode CreateSectionForDMAndOperator(DM dm, Operator *op, PetscSection *sec) {
  PetscFunctionBeginUser;
  PetscInt n_field          = 1;
  PetscInt n_field_comps[1] = {op->num_components};
  char   **comp_names       = op->field_names;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));

  PetscCall(PetscSectionCreate(comm, sec));
  PetscCall(PetscSectionSetNumFields(*sec, n_field));

  PetscInt n_field_dof_tot = 0;
  for (PetscInt f = 0; f < n_field; ++f) {
    PetscCall(PetscSectionSetFieldComponents(*sec, f, n_field_comps[f]));
    for (PetscInt c = 0; c < n_field_comps[f]; ++c, ++n_field_dof_tot) {
      PetscCall(PetscSectionSetComponentName(*sec, f, c, comp_names[c]));
    }
  }

  // set the number of degrees of freedom in each cell
  PetscInt c_start, c_end;  // starting and ending cell points
  PetscCall(DMPlexGetHeightStratum(dm, 0, &c_start, &c_end));
  PetscCall(PetscSectionSetChart(*sec, c_start, c_end));
  for (PetscInt c = c_start; c < c_end; ++c) {
    for (PetscInt f = 0; f < n_field; ++f) {
      PetscCall(PetscSectionSetFieldDof(*sec, c, f, n_field_comps[f]));
    }
    PetscCall(PetscSectionSetDof(*sec, c, n_field_dof_tot));
  }
  PetscCall(PetscSectionSetUp(*sec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Defines the distinct boundaries (edges) for the operator.
PetscErrorCode SetOperatorBoundaries(Operator *op, PetscInt num_boundaries, RDyBoundary *boundaries) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCheck(num_boundaries > 0, comm, PETSC_ERR_USER, "Number of operator boundaries must be positive");
  PetscCheck(boundaries, comm, PETSC_ERR_USER, "Operator boundary array must be non-NULL");

  PetscCall(PetscCalloc1(num_boundaries, &op->boundaries));
  memcpy(op->boundaries, boundaries, sizeof(RDyBoundary) * num_boundaries);

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Defines the distinct regions of owned cells for the operator.
PetscErrorCode SetOperatorRegions(Operator *op, PetscInt num_regions, RDyRegion *regions) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCheck(num_regions > 0, comm, PETSC_ERR_USER, "Number of operator regions must be positive");
  PetscCheck(regions, comm, PETSC_ERR_USER, "Operator region array must be non-NULL");

  PetscCall(PetscCalloc1(num_regions, &op->regions));
  memcpy(op->regions, regions, sizeof(RDyRegion) * num_regions);

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Defines the computational domain for the operator.
PetscErrorCode SetOperatorDomain(Operator *op, DM dm, RDyMesh *mesh) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCheck(mesh, comm, PETSC_ERR_USER, "Operator mesh must be non-NULL");

  op->dm   = dm;
  op->mesh = mesh;

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ReadyCeedOperator(Operator *op) {
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ReadyPetscOperator(Operator *op) {
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// performs all work necessary to make the operator ready for use
PetscErrorCode ReadyOperator(Operator *op) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));

  // make sure preconditions are met
  PetscCheck(op->dm, comm, PETSC_ERR_USER, "Cannot ready an operator with no associated DM");
  PetscCheck(op->num_boundaries > 0, comm, PETSC_ERR_USER, "Cannot ready an operator with no boundaries");
  PetscCheck(op->num_regions > 0, comm, PETSC_ERR_USER, "Cannot ready an operator with no regions");
  PetscCheck(op->mesh, comm, PETSC_ERR_USER, "Cannot ready an operator with no mesh");

  if (CeedEnabled()) {
    PetscCall(ReadyCeedOperator(op));
  } else {
    PetscCall(ReadyPetscOperator(op));
  }

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

PetscErrorCode GetLocalOperatorCourantNumberDiagnostics(Operator *op, CourantNumberDiagnostics *diags) {
  PetscFunctionBegin;
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
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RestoreOperatorMaterialProperty(Operator *op, RDyRegion region, OperatorMaterialPropertyId property_id, OperatorData *property) {
  PetscFunctionBegin;

  switch (property_id) {
    case OPERATOR_MANNINGS:
      if (CeedEnabled()) {
        PetscCall(RestoreCeedOperatorRegionData(op, region, material_property_names[property_id], property));
      } else {
        PetscCall(RestorePetscOperatorRegionData(op, region, op->petsc.material_properties[region.index][property_id], property));
      }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
