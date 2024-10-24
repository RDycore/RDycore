#include <ceed/ceed.h>
#include <petscdmceed.h>
#include <private/rdycoreimpl.h>
#include <private/rdysweimpl.h>

//-------------------------
// CEED context management
//-------------------------

// global CEED resource name and context
static char ceed_resource[PETSC_MAX_PATH_LEN + 1] = {0};
static Ceed ceed_context;

/// returns true iff CEED is enabled
PetscBool CeedEnabled(void) { return (ceed_resource[0]) ? PETSC_TRUE : PETSC_FALSE; }

/// returns the global CEED context, which is only valid if CeedEnabled()
/// returns PETSC_TRUE
Ceed CeedContext(void) { return ceed_context; }

/// retrieves the appropriate PETSc Vec type for the selected CEED backend
PetscErrorCode GetCeedVecType(VecType *vec_type) {
  PetscFunctionBegin;

  CeedMemType mem_type_backend;
  PetscCallCEED(CeedGetPreferredMemType(ceed_context, &mem_type_backend));
  switch (mem_type_backend) {
    case CEED_MEM_HOST:
      *vec_type = VECSTANDARD;
      break;
    case CEED_MEM_DEVICE: {
      const char *resolved;
      PetscCallCEED(CeedGetResource(ceed_context, &resolved));
      if (strstr(resolved, "/gpu/cuda")) *vec_type = VECCUDA;
      else if (strstr(resolved, "/gpu/hip")) *vec_type = VECKOKKOS;
      else if (strstr(resolved, "/gpu/sycl")) *vec_type = VECKOKKOS;
      else *vec_type = VECSTANDARD;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Sets the CEED resource string to the given string, initializing the global
/// CEED context if the argument is specified. If CEED has already been enabled
/// with a different resource, a call to this function deletes the global
/// context and recreates it. If the resource string is empty or NULL, CEED is
/// disabled.
/// @param resource a CEED resource string, possibly empty or NULL
PetscErrorCode SetCeedResource(char *resource) {
  PetscFunctionBegin;
  if (ceed_resource[0]) {  // we already have a context
    CeedDestroy(&ceed_context);
  }
  if (resource && resource[0]) {
    strncpy(ceed_resource, resource, PETSC_MAX_PATH_LEN);
    PetscCallCEED(CeedInit(ceed_resource, &ceed_context));
  } else {
    ceed_resource[0] = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

//-----------------------
// Debugging diagnostics
//-----------------------

// MPI datatype corresponding to CourantNumberDiagnostics. Created during
// CreateSWEOperator.
static MPI_Datatype courant_num_diags_type;

// MPI operator used to determine the prevailing diagnostics for the maximum
// courant number on all processes. Created during CreateSWEOperator.
static MPI_Op courant_num_diags_op;

// function backing the above MPI operator
static void FindCourantNumberDiagnostics(void *in_vec, void *result_vec, int *len, MPI_Datatype *type) {
  CourantNumberDiagnostics *in_diags     = in_vec;
  CourantNumberDiagnostics *result_diags = result_vec;

  // select the item with the maximum courant number
  for (int i = 0; i < *len; ++i) {
    if (in_diags[i].max_courant_num > result_diags[i].max_courant_num) {
      result_diags[i] = in_diags[i];
    }
  }
}

// this function destroys the MPI type and operator associated with CourantNumberDiagnostics
static void DestroyCourantNumberDiagnostics(void) {
  MPI_Op_free(&courant_num_diags_op);
  MPI_Type_free(&courant_num_diags_type);
}
// this function initializes some MPI machinery for the above Courant number
// diagnostics, and is called by CreateBasicOperator, which is called when a
// CEED or PETSc Operator is created
static PetscErrorCode InitCourantNumberDiagnostics(void) {
  PetscFunctionBegin;

  static PetscBool initialized = PETSC_FALSE;

  if (!initialized) {
    // create an MPI data type for the CourantNumberDiagnostics struct
    const int      num_blocks             = 4;
    const int      block_lengths[4]       = {1, 1, 1, 1};
    const MPI_Aint block_displacements[4] = {
        offsetof(CourantNumberDiagnostics, max_courant_num),
        offsetof(CourantNumberDiagnostics, global_edge_id),
        offsetof(CourantNumberDiagnostics, global_cell_id),
        offsetof(CourantNumberDiagnostics, is_set),
    };
    MPI_Datatype block_types[4] = {MPIU_REAL, MPI_INT, MPI_INT, MPIU_BOOL};
    MPI_Type_create_struct(num_blocks, block_lengths, block_displacements, block_types, &courant_num_diags_type);
    MPI_Type_commit(&courant_num_diags_type);

    // create a corresponding reduction operator for the new type
    MPI_Op_create(FindCourantNumberDiagnostics, 1, &courant_num_diags_op);

    // make sure the operator and the type are destroyed upon exit
    PetscCall(RDyOnFinalize(DestroyCourantNumberDiagnostics));

    initialized = PETSC_TRUE;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// CEED uses C99 VLA features for shaping multidimensional
// arrays, which don't have the same drawbacks as VLA allocations.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla"

// these are used to register CEED operator events for profiling
static PetscClassId RDY_CLASSID;
PetscLogEvent       RDY_CeedOperatorApply;

/// Creates a PetscSection appropriate for the selected system.
/// FIXME: this belongs somewhere else
PetscErrorCode CreateSection(RDy rdy, PetscSection *section) {
  PetscFunctionBegin;

  // for now, pass the call to CreateSWESection
  PetscCheck(rdy->config.physics.flow.mode == FLOW_SWE, rdy->comm, PETSC_ERR_USER, "Only the 'swe' flow mode is currently supported.");
  PetscCall(CreateSWESection(rdy, section));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateBasicOperator(DM dm, RDyMesh *mesh, PetscInt num_components, PetscInt num_boundaries, RDyBoundary *boundaries,
                                          Operator *op) {
  PetscFunctionBegin;

  PetscCall(InitCourantNumberDiagnostics());

  op->dm             = dm;
  op->mesh           = mesh;
  op->num_components = num_components;
  op->num_boundaries = num_boundaries;
  op->boundaries     = boundaries;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Creates an operator implemented by CEED flux and source operators for a
/// domain represented by the given mesh and boundaries.
///
/// Such an operator is supported by underlying CeedOperator objects representing
///
/// * the calculation of fluxes between cells
/// * the accumulation of source terms
///
/// This function creates an "empty" operator that must be assembled by subsequent
/// function calls in a specific order:
///
/// To set up the calculation of inter-cell fluxes:
///
/// 1. Call AddCeedInteriorFluxOperator with a CeedOperator that computes fluxes
///    between cells in the interior of the domain.
/// 2. For each boundery, call AddCeedBoundaryFluxOperator with the boundary's
///    index (0 through num_boundaries-1, in order) and a CeedOperator that
///    computes fluxes between cells on that boundary.
///
/// This means the flux operator has 1 + num_boundaries CEED sub-operators.
///
/// To set up sources:
/// * Call AddCeedSourceOperator with a CeedOperator that computes or
///   accumulates source terms for each cell in the domain.
///
/// The source operator can have one or more CEED sub-operators.
///
/// @param [in]  dm the DM associated with the solution vector
/// @param [in]  mesh a mesh defining the computational domain
/// @param [in]  num_components the number of components in the solution vector
/// @param [in]  num_boundaries the number of boundaries in the computational domain
/// @param [in]  boundaries a pointer to an array of boundaries
/// @param [out] op a newly created "empty" operator, ready for assembly
PetscErrorCode CreateCeedOperator(DM dm, RDyMesh *mesh, PetscInt num_components, PetscInt num_boundaries, RDyBoundary *boundaries, Operator *op) {
  PetscFunctionBegin;

  // register a logging event for applying our CEED operator
  PetscCall(PetscClassIdRegister("RDycore", &RDY_CLASSID));
  PetscCall(PetscLogEventRegister("CeedOperatorApp", RDY_CLASSID, &RDY_CeedOperatorApply));

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));

  PetscCheck(CeedEnabled(), comm, PETSC_ERR_USER, "Can't create a CEED operator: CEED is disabled");
  PetscCall(CreateBasicOperator(dm, mesh, num_components, num_boundaries, boundaries, op));

  // a CEED flux operator is a composite operator consisting of 1 + op->num_boundaries
  // sub-operators: 1 for the interior of the domain and the rest for its boundaries
  PetscCallCEED(CeedCompositeOperatorCreate(ceed_context, &op->ceed.flux_operator));

  // a CEED source operator is a composite operator consisting of 1 or more sub-operators
  // that handle source terms
  PetscCallCEED(CeedCompositeOperatorCreate(ceed_context, &op->ceed.source_operator));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Adds a sub-operator that computes fluxes between interior cells to the given
// Operator. This should be the first CEED flux sub-operator added to the
// Operator. The sub-operator is consumed in the process.
//
// The following input fields should be present:
//
// 1. "geom"    - a set of geometric parameters
// 2. "q_left"  - the state in the "left" cell of an edge for each flux computed
// 3. "q_right" - the state in the "right" cell of an edge for each flux computed
//
// The following output fields should be present:
//
// 1. "geom"           - a set of computed geometric parameters
// 2. "cell_left"      - the left state corresponding to the Riemann flux
// 3. "cell_right"     - the right state corresponding to the Riemann flux
// 4. "flux"           - the Riemann flux through the edge separating left and right cells
// 5. "courant_number" - the Courant numbers for the x and y velocities
//
// @param op               [in] the operator to which the interior flux sub-operator is added
// @param interior_flux_op [in] the sub-operator computing the interior fluxes
PetscErrorCode AddCeedInteriorFluxSubOperator(Operator *op, CeedOperator interior_flux_op) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCheck(CeedEnabled(), comm, PETSC_ERR_USER, "Can't add a CEED sub-interior flux operator: CEED is disabled");

  // this should be the first sub-operator added
  CeedInt num_sub_operators;
  PetscCallCEED(CeedCompositeOperatorGetNumSub(op->ceed.flux_operator, &num_sub_operators));
  PetscCheck(num_sub_operators == 0, comm, PETSC_ERR_USER, "CEED interior flux sub-operator must be the first sub-operator added");

  // check for required input, output fields
  const char *required_inputs[] = {
      "geom",
      "q_left",
      "q_right",
      NULL,
  };
  for (PetscInt f = 0; required_inputs[f]; ++f) {
    CeedOperatorField field;
    PetscCallCEED(CeedOperatorGetFieldByName(interior_flux_op, required_inputs[f], &field));
    PetscCheck(field, comm, PETSC_ERR_USER, "CEED interior flux sub-operator missing required input field: %s", required_inputs[f]);
  }
  const char *required_outputs[] = {
      "geom", "cell_left", "cell_right", "flux", "courant_number", NULL,
  };
  for (PetscInt f = 0; required_outputs[f]; ++f) {
    CeedOperatorField field;
    PetscCallCEED(CeedOperatorGetFieldByName(interior_flux_op, required_outputs[f], &field));
    PetscCheck(field, comm, PETSC_ERR_USER, "CEED interior flux sub-operator missing required output field: %s", required_outputs[f]);
  }

  PetscCallCEED(CeedCompositeOperatorAddSub(op->ceed.flux_operator, interior_flux_op));
  PetscCallCEED(CeedOperatorDestroy(&interior_flux_op));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Retrieves the interior flux sub-operator added to the operator by a prior
/// call to AddCeedInteriorFluxOperator(), throwing an error if no such call was
/// made.
PetscErrorCode GetCeedInteriorFluxSubOperator(Operator *op, CeedOperator *interior_flux_op) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCheck(CeedEnabled(), comm, PETSC_ERR_USER, "Can't get CEED interior flux sub-operator: CEED is disabled");

  CeedInt num_sub_operators;
  PetscCallCEED(CeedCompositeOperatorGetNumSub(op->ceed.flux_operator, &num_sub_operators));
  PetscCheck(num_sub_operators >= 1, comm, PETSC_ERR_USER, "Can't get CEED interior flux sub-operator: no operator was assigned!");
  CeedOperator *sub_operators;
  PetscCallCEED(CeedCompositeOperatorGetSubList(op->ceed.flux_operator, &sub_operators));

  *interior_flux_op = sub_operators[0];
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Adds a sub-operator that computes fluxes on boundary cells to the given
// Operator. This should be called once for each boundary in the domain, and
// it should be called after the interior flux sub-operator has been added to
// the Operator. The sub-operator is consumed in the process.
//
// The following input fields should be present:
//
// 1. "geom"    - a set of geometric parameters
// 2. "q_left"  - the state in the "left" cell of an edge for each flux computed
//
// The following output fields should be present:
//
// 1. "geom"           - a set of computed geometric parameters
// 2. "cell_left"      - the left state corresponding to the Riemann flux
// 4. "flux"           - the Riemann flux through the edge separating left and right cells
// 5. "courant_number" - the Courant numbers for the x and y velocities
//
// @param op               [in] the operator to which the boundary flux sub-operator is added
// @param boundary         [in] the boundary associated with the sub-operator
// @param interior_flux_op [in] the sub-operator computing the boundary fluxes
PetscErrorCode AddCeedBoundaryFluxSubOperator(Operator *op, RDyBoundary boundary, CeedOperator boundary_flux_op) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCheck(CeedEnabled(), comm, PETSC_ERR_USER, "Can't add a CEED boundary flux operator: CEED is disabled");

  // check the boundary index
  PetscCheck(boundary.index >= 0, comm, PETSC_ERR_USER, "Can't add a CEED boundary flux operator: negative boundary index %" PetscInt_FMT,
             boundary.index);
  CeedInt num_sub_operators;
  PetscCallCEED(CeedCompositeOperatorGetNumSub(op->ceed.flux_operator, &num_sub_operators));
  PetscCheck(boundary.index == num_sub_operators - 1, comm, PETSC_ERR_USER,
             "Invalid CEED boundary flux sub-operator index: %" PetscInt_FMT " (should be %" PetscInt_FMT ")", boundary.index, num_sub_operators - 1);

  // check for required input, output fields
  const char *required_inputs[] = {
      "geom",
      "q_left",
      NULL,
  };
  for (PetscInt f = 0; required_inputs[f]; ++f) {
    CeedOperatorField field;
    PetscCallCEED(CeedOperatorGetFieldByName(boundary_flux_op, required_inputs[f], &field));
    PetscCheck(field, comm, PETSC_ERR_USER, "CEED boundary flux sub-operator missing required input field: %s", required_inputs[f]);
  }
  const char *required_outputs[] = {
      "geom", "cell_left", "flux", "courant_number", NULL,
  };
  for (PetscInt f = 0; required_outputs[f]; ++f) {
    CeedOperatorField field;
    PetscCallCEED(CeedOperatorGetFieldByName(boundary_flux_op, required_outputs[f], &field));
    PetscCheck(field, comm, PETSC_ERR_USER, "CEED boundary flux sub-operator missing required output field: %s", required_outputs[f]);
  }

  PetscCallCEED(CeedCompositeOperatorAddSub(op->ceed.flux_operator, boundary_flux_op));
  PetscCallCEED(CeedOperatorDestroy(&boundary_flux_op));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Retrieves a boundary flux sub-operator added to the operator by a prior
/// call to AddCeedBoundaryFluxOperator(), throwing an error if no such call was
/// made.
PetscErrorCode GetCeedBoundaryFluxSubOperator(Operator *op, RDyBoundary boundary, CeedOperator *boundary_flux_op) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCheck(CeedEnabled(), comm, PETSC_ERR_USER, "Can't get CEED boundary flux sub-operator: CEED is disabled");
  PetscCheck(boundary.index >= 0 && boundary.index < op->num_boundaries, comm, PETSC_ERR_USER,
             "Can't get CEED boundary flux operator: invalid boundary index (%" PetscInt_FMT ")", boundary.index);

  CeedInt num_sub_operators;
  PetscCallCEED(CeedCompositeOperatorGetNumSub(op->ceed.flux_operator, &num_sub_operators));
  PetscCheck(num_sub_operators >= 1 + boundary.index, comm, PETSC_ERR_USER,
             "Can't find CEED boundary flux sub-operator with boundary index %" PetscInt_FMT, boundary.index);
  CeedOperator *sub_operators;
  PetscCallCEED(CeedCompositeOperatorGetSubList(op->ceed.flux_operator, &sub_operators));

  *boundary_flux_op = sub_operators[1 + boundary.index];
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Adds a sub-operator that computes source terms on all cells in the
// computational domain to the given Operator. This can be called to add as
// many source sub-operators as necessary to implement the Operator. The sub-
// operator is consumed in the process.
//
// The following input fields should be present:
//
// 1. "geom"     - a set of geometric parameters
// 2. "swe_src"  - source data already present in the cells
// 3. "riemannf" - the Riemann flux divergence stored in the cells
//
// The following output fields should be present:
//
// 1. "cell" - the accumulated sources in each cell
//
// @param op               [in] the operator to which the boundary flux sub-operator is added
// @param boundary         [in] the boundary associated with the sub-operator
// @param interior_flux_op [in] the sub-operator computing the boundary fluxes
PetscErrorCode AddCeedSourceSubOperator(Operator *op, CeedOperator source_op) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCheck(CeedEnabled(), comm, PETSC_ERR_USER, "Can't add a CEED source sub-operator: CEED is disabled");

  // check for required input, output fields
  const char *required_inputs[] = {
      "geom",       "swe_src",
      "mannings_n",  // FIXME: SWE-specific!
      "riemannf",
      "q",  // FIXME: SWE-specific!
      NULL,
  };
  for (PetscInt f = 0; required_inputs[f]; ++f) {
    CeedOperatorField field;
    PetscCallCEED(CeedOperatorGetFieldByName(source_op, required_inputs[f], &field));
    PetscCheck(field, comm, PETSC_ERR_USER, "CEED source sub-operator missing required input field: %s", required_inputs[f]);
  }
  const char *required_outputs[] = {
      "cell",
      NULL,
  };
  for (PetscInt f = 0; required_outputs[f]; ++f) {
    CeedOperatorField field;
    PetscCallCEED(CeedOperatorGetFieldByName(source_op, required_outputs[f], &field));
    PetscCheck(field, comm, PETSC_ERR_USER, "CEED source sub-operator missing required output field: %s", required_outputs[f]);
  }

  PetscCallCEED(CeedCompositeOperatorAddSub(op->ceed.source_operator, source_op));
  PetscCallCEED(CeedOperatorDestroy(&source_op));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Retrieves the interior flux CeedOperator added to the operator by a prior
/// call to AddCeedSourceOperator(), throwing an error if no such call was
/// made.
PetscErrorCode GetCeedSourceSubOperator(Operator *op, CeedOperator *source_op) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCheck(CeedEnabled(), comm, PETSC_ERR_USER, "Can't get CEED source sub-operator: CEED is disabled");

  CeedInt num_sub_operators;
  PetscCallCEED(CeedCompositeOperatorGetNumSub(op->ceed.source_operator, &num_sub_operators));
  PetscCheck(num_sub_operators >= 1, comm, PETSC_ERR_USER, "Can't get CEED source sub-operator: no operator was assigned!");
  CeedOperator *sub_operators;
  PetscCallCEED(CeedCompositeOperatorGetSubList(op->ceed.source_operator, &sub_operators));

  // FIXME: this assumes the existence of only 1 source sub-operator (not enforced!)
  *source_op = sub_operators[1 + op->num_boundaries];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Creates an operator implemented by PETSc functions for a domain represented
/// by the given mesh and boundaries.
///
/// @param [in]  dm the DM associated with the solution vector
/// @param [in]  mesh a mesh defining the computational domain
/// @param [in]  num_components the number of components in the solution vector
/// @param [in]  num_boundaries the number of boundaries in the computational domain
/// @param [in]  boundaries a pointer to an array of boundaries
/// @param [out] op a newly created operator
PetscErrorCode CreatePetscOperator(DM dm, RDyMesh *mesh, PetscInt num_components, PetscInt num_boundaries, RDyBoundary *boundaries, Operator *op) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCheck(!CeedEnabled(), comm, PETSC_ERR_USER, "Can't create a PETSc operator: CEED is enabled");

  PetscCall(CreateBasicOperator(dm, mesh, num_components, num_boundaries, boundaries, op));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Destroys the given operator, freeing its resources.
PetscErrorCode DestroyOperator(Operator *op) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));

  // is anyone manipulating boundary data, source data, etc?
  if (op->lock.boundary_data) {
    for (PetscInt i = 0; i < op->num_boundaries; ++i) {
      PetscCheck(op->lock.boundary_data[i] == NULL, comm, PETSC_ERR_USER, "Could not destroy operator: boundary data is in use");
    }
  }
  PetscCheck(op->lock.source_data == NULL, comm, PETSC_ERR_USER, "Could not destroy operator: source data is in use");

  PetscFree(op->lock.boundary_data);

  PetscBool ceed_enabled = CeedEnabled();

  if (op->petsc.context) {
    // FIXME: SWE-specific logic!
    PetscCall(DestroyPetscSWEFlux(op->petsc.context, ceed_enabled, op->num_boundaries));
  }

  if (ceed_enabled) {
    PetscCallCEED(CeedOperatorDestroy(&op->ceed.flux_operator));
    PetscCallCEED(CeedOperatorDestroy(&op->ceed.source_operator));
    PetscCallCEED(CeedVectorDestroy(&op->ceed.u_local));
    PetscCallCEED(CeedVectorDestroy(&op->ceed.rhs));
    PetscCallCEED(CeedVectorDestroy(&op->ceed.sources));
  }

  *op = (Operator){0};

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetCeedOperatorTimeStep(CeedOperator op, PetscReal dt) {
  PetscFunctionBeginUser;

  // CEED operators store the time step under the "time step" label
  CeedContextFieldLabel label;
  PetscCallCEED(CeedOperatorGetContextFieldLabel(op, "time step", &label));
  PetscCallCEED(CeedOperatorSetContextDouble(op, label, &dt));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetOperatorTimeStep(Operator *op, PetscReal dt) {
  PetscFunctionBegin;
  if (CeedEnabled()) {
    PetscCall(SetCeedOperatorTimeStep(op->ceed.flux_operator, dt));
    PetscCall(SetCeedOperatorTimeStep(op->ceed.source_operator, dt));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline CeedMemType MemTypeP2C(PetscMemType mem_type) { return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST; }

static PetscErrorCode ApplyCeedOperator(Operator *op, PetscReal t, Vec u, Vec dudt) {
  PetscFunctionBegin;
  {
    // The computation of fluxes across internal and boundary edges via CeedOperator is done
    // in the following three stages:
    //
    // a) pre-CeedOperatorApply stage
    //    - point a CeedVector (u_local_ceed) at the PETSc Vec (u)
    //    - get an appropriate PETSc Vec from the DM (F_local), and point
    //      the CeedVector f_ceed at it
    //
    // b) CeedOperatorApply stage
    //    - apply the CEED flux operator(s), transforming u_local_ceed -> f_ceed
    //
    // c) post-CeedOperatorApply stage
    //    - accumulate values from F_local to dudt via local-to-global scatter
    //    - clean up

    PetscScalar *u_local, *f;
    PetscMemType mem_type;
    Vec          F_local;

    CeedVector u_local_ceed = op->ceed.u_local;
    CeedVector f_ceed       = op->ceed.rhs;

    // 1. Sets the pointer of a CeedVector to a PETSc Vec: u_local_ceed --> U_local
    PetscCall(VecGetArrayAndMemType(u, &u_local, &mem_type));
    PetscCallCEED(CeedVectorSetArray(u_local_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, u_local));

    // 2. Sets the pointer of a CeedVector to a PETSc Vec: f_ceed --> F_local
    PetscCall(DMGetLocalVector(op->dm, &F_local));
    PetscCall(VecGetArrayAndMemType(F_local, &f, &mem_type));
    PetscCallCEED(CeedVectorSetArray(f_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, f));

    // 3. Apply the CeedOperator associated with the internal and boundary edges
    PetscCall(PetscLogEventBegin(RDY_CeedOperatorApply, u, dudt, 0, 0));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCEED(CeedOperatorApply(op->ceed.flux_operator, u_local_ceed, f_ceed, CEED_REQUEST_IMMEDIATE));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogEventEnd(RDY_CeedOperatorApply, u, dudt, 0, 0));

    // 4. Resets memory pointer of CeedVectors
    PetscCallCEED(CeedVectorTakeArray(f_ceed, MemTypeP2C(mem_type), &f));
    PetscCallCEED(CeedVectorTakeArray(u_local_ceed, MemTypeP2C(mem_type), &u_local));

    // 5. Restore pointers to the PETSc Vecs
    PetscCall(VecRestoreArrayAndMemType(F_local, &f));
    PetscCall(VecRestoreArrayAndMemType(u, &u_local));

    // 6. Zero out values in du/dt and then add F_local to F via local-to-global scatter
    PetscCall(VecZeroEntries(dudt));
    PetscCall(DMLocalToGlobal(op->dm, F_local, ADD_VALUES, dudt));

    // 7. Restore the F_local
    PetscCall(DMRestoreLocalVector(op->dm, &F_local));
  }

  {
    // The computation of contribution of the source-sink term via CeedOperator is done
    // in the following three stages:
    //
    // a) Pre-CeedOperatorApply stage:
    //    - Set memory pointer of a CeedVector (u_local_ceed) to PETSc Vec (U_local)
    //    - A copy of the PETSc Vec F is made as flux_divergences. Then, memory pointer of a CeedVector (riemannf_ceed)
    //      to flux_divergences.
    //    - Set memory pointer of a CeedVector (s_ceed) to PETSc Vec (F)
    //
    // b) CeedOperatorApply stage
    //    - Apply the CeedOparator in which u_local_ceed is an input, while s_ceed is an output.
    //
    // c) Post-CeedOperatorApply stage:
    //    - Clean up memory

    // 1. Get the CeedVector associated with the "riemannf" CeedOperatorField
    CeedOperatorField riemannf_field;
    CeedVector        riemannf_ceed;
    SWESourceOperatorGetRiemannFlux(op, &riemannf_field);
    PetscCallCEED(CeedOperatorFieldGetVector(riemannf_field, &riemannf_ceed));

    PetscScalar *u_ptr, *f, *flux_divergences;
    PetscMemType mem_type;
    CeedVector   u_local_ceed = op->ceed.u_local;
    CeedVector   s_ceed       = op->ceed.sources;

    // 2. Sets the pointer of a CeedVector to a PETSc Vec: u_local_ceed --> U_local
    PetscCall(VecGetArrayAndMemType(u, &u_ptr, &mem_type));
    PetscCallCEED(CeedVectorSetArray(u_local_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, u_ptr));

    // 3. Make a duplicate copy of the F as the values will be used as input for the CeedOperator
    //    corresponding to the source-sink term
    PetscCall(VecCopy(dudt, op->ceed.flux_divergences));

    // 4. Sets the pointer of a CeedVector to a PETSc Vec: flux_divergences --> riemannf_ceed
    PetscCall(VecGetArrayAndMemType(op->ceed.flux_divergences, &flux_divergences, &mem_type));
    PetscCallCEED(CeedVectorSetArray(riemannf_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, flux_divergences));

    // 5. Sets the pointer of a CeedVector to a PETSc Vec: F --> s_ceed
    PetscCall(VecGetArrayAndMemType(dudt, &f, &mem_type));
    PetscCallCEED(CeedVectorSetArray(s_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, f));

    // 6. Apply the source CeedOperator
    PetscCall(PetscLogEventBegin(RDY_CeedOperatorApply, u, dudt, 0, 0));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCEED(CeedOperatorApply(op->ceed.source_operator, u_local_ceed, s_ceed, CEED_REQUEST_IMMEDIATE));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogEventEnd(RDY_CeedOperatorApply, u, dudt, 0, 0));

    // 7. Reset memory pointer of CeedVectors
    PetscCallCEED(CeedVectorTakeArray(s_ceed, MemTypeP2C(mem_type), &f));
    PetscCallCEED(CeedVectorTakeArray(riemannf_ceed, MemTypeP2C(mem_type), &flux_divergences));
    PetscCallCEED(CeedVectorTakeArray(u_local_ceed, MemTypeP2C(mem_type), &u_ptr));

    // 8. Restore pointers to the PETSc Vecs
    PetscCall(VecRestoreArrayAndMemType(u, &u_ptr));
    PetscCall(VecRestoreArrayAndMemType(dudt, &f));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Applies the given operator to the local solution vector at the given time,
/// computing and storing its time derivative.
/// \param [inout] op   the operator
/// \param [in]    t    the time at which the operator is applied
/// \param [in]    u    the local solution vector
/// \param [out]   dudt the locally right hand side storing the computed time derivative of u
PetscErrorCode ApplyOperator(Operator *op, PetscReal t, Vec u, Vec dudt) {
  PetscFunctionBegin;

  // reset courant number diagnostics
  op->courant_diags = (CourantNumberDiagnostics){
      .max_courant_num = 0.0,
      .global_edge_id  = -1,
      .global_cell_id  = -1,
      .is_set          = PETSC_FALSE,
  };

  if (CeedEnabled()) {
    PetscCall(ApplyCeedOperator(op, t, u, dudt));
  } else {
    PetscCall(op->petsc.apply(op->petsc.rdy, t, u, dudt));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Updates the Courant number diagnostics with information about the most
/// recent application of the operator.
/// \param [inout] op    the operator
PetscErrorCode UpdateOperatorCourantNumberDiagnostics(Operator *op) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));

  // update diagnostics locally (if needed)
  if (op->update_local_courant_diags) {
    PetscCall(op->update_local_courant_diags(op, &op->courant_diags));
  }

  // reduce and set updated flag
  MPI_Allreduce(MPI_IN_PLACE, &op->courant_diags, 1, courant_num_diags_type, courant_num_diags_op, comm);
  op->courant_diags.is_set = PETSC_TRUE;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Retrieves information about the conditions under which the maximum Courant
/// was encountered in the last call to UpdateOperatorCourantNumberDiagnostics.
/// \param [inout] op    the operator
/// \param [out]   diags information about the maximum Courant number
PetscErrorCode GetOperatorCourantNumberDiagnostics(Operator *op, CourantNumberDiagnostics *diags) {
  PetscFunctionBegin;

  *diags = op->courant_diags;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// acquires exclusive access to boundary data for flux operators
PetscErrorCode GetOperatorBoundaryData(Operator *op, RDyBoundary boundary, OperatorBoundaryData *boundary_data) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));

  *boundary_data = (OperatorBoundaryData){0};
  if (!op->lock.boundary_data) {
    PetscCall(PetscCalloc1(op->num_boundaries, &op->lock.boundary_data));
  }
  PetscCheck(boundary.index >= 0 && boundary.index < op->num_boundaries, comm, PETSC_ERR_USER,
             "Invalid boundary for operator boundary data (index: %" PetscInt_FMT ")", boundary.index);
  PetscCheck(!op->lock.boundary_data[boundary.index], comm, PETSC_ERR_USER,
             "Could not acquire lock on operator boundary data -- another entity has access");
  op->lock.boundary_data[boundary.index] = boundary_data;
  boundary_data->op                      = op;
  boundary_data->boundary                = boundary;
  boundary_data->num_components          = op->num_components;

  if (CeedEnabled()) {
    // get the relevant boundary sub-operator
    CeedOperator *sub_ops;
    PetscCallCEED(CeedCompositeOperatorGetSubList(op->ceed.flux_operator, &sub_ops));
    CeedOperator flux_op = sub_ops[1 + boundary_data->boundary.index];

    // fetch the relevant vectors
    CeedOperatorField field;
    PetscCallCEED(CeedOperatorGetFieldByName(flux_op, "q_dirichlet", &field));
    PetscCallCEED(CeedOperatorFieldGetVector(field, &boundary_data->values.ceed.vec));

    PetscCallCEED(CeedOperatorGetFieldByName(flux_op, "flux", &field));
    PetscCallCEED(CeedOperatorFieldGetVector(field, &boundary_data->fluxes.ceed.vec));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// sets boundary values for the given component on the boundary associated with
// the given operator boundary data
PetscErrorCode SetOperatorBoundaryValues(OperatorBoundaryData *boundary_data, PetscInt component, PetscReal *boundary_values) {
  PetscFunctionBegin;

  if (CeedEnabled()) {
    // if this is the first update, get access to the vector's data
    if (!boundary_data->values.updated) {
      PetscCallCEED(CeedVectorGetArray(boundary_data->values.ceed.vec, CEED_MEM_HOST, &boundary_data->values.ceed.data));
      boundary_data->values.updated = PETSC_TRUE;
    }

    // reshape for multicomponent access
    CeedScalar(*values)[boundary_data->num_components];
    *((CeedScalar **)&values) = boundary_data->values.ceed.data;

    // set the data
    for (CeedInt e = 0; e < boundary_data->boundary.num_edges; ++e) {
      values[e][component] = boundary_values[e];
    }
  } else {
    // if this is the first update, get access to the vector's data
    if (!boundary_data->values.updated) {
      PetscCall(VecGetArray(boundary_data->values.petsc.vec, &boundary_data->values.petsc.data));
      boundary_data->values.updated = PETSC_TRUE;
    }

    /* FIXME: eventually, we can have something like this vvv, but for now we're
     * FIXME: SWE-specific
    // set the data
    PetscReal *u = boundary_data->storage.petsc.data;
    for (PetscInt e = 0; e < boundary_data->boundary.num_edges; ++e) {
      u[e * boundary_data->num_components + component] = boundary_values[e];
    }
    */

    Operator   *op       = boundary_data->op;
    RDyBoundary boundary = boundary_data->boundary;

    // fetch the boundary data
    RiemannDataSWE bdata;
    PetscCall(GetPetscSWEDirichletBoundaryValues(op->petsc.context, boundary.index, &bdata));

    // set the boundary values
    RDyCells  *cells  = &op->mesh->cells;
    RDyEdges  *edges  = &op->mesh->edges;
    PetscReal *values = (component == 0) ? bdata.h : (component == 1) ? bdata.hu : bdata.hv;
    for (PetscInt e = 0; e < boundary.num_edges; ++e) {
      PetscInt iedge = boundary.edge_ids[e];
      PetscInt icell = edges->cell_ids[2 * iedge];
      if (cells->is_local[icell]) {
        values[e] = boundary_values[3 * e + component];
      }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Copies boundary flux data for the selected component into the supplied
/// array.
/// NOTE: if you call this function, the subsequent call to RestoreOperatorBoundaryData()
/// NOTE: will zero boundary fluxes.
/// @param [in]  boundary_data boundary data associated with a given operator/boundary
/// @param [in]  component the component of interest
/// @param [out] data      the array to which the flux (edge) data is copied
PetscErrorCode GetOperatorBoundaryFluxes(OperatorBoundaryData *boundary_data, PetscInt component, PetscReal *fluxes) {
  PetscFunctionBegin;

  if (CeedEnabled()) {
    // if this is the first fetch, get access to the vector's data and mark it
    // to be zeroed upon restoration
    if (!boundary_data->fluxes.updated) {
      PetscCallCEED(CeedVectorGetArray(boundary_data->fluxes.ceed.vec, CEED_MEM_HOST, &boundary_data->fluxes.ceed.data));
      boundary_data->fluxes.updated = PETSC_TRUE;
    }

    // reshape for multicomponent access
    CeedScalar(*bfluxes)[boundary_data->num_components];
    *((CeedScalar **)&bfluxes) = boundary_data->fluxes.ceed.data;

    for (PetscInt e = 0; e < boundary_data->boundary.num_edges; ++e) {
      fluxes[e] = bfluxes[component][e];
    }
  } else {
    // FIXME: need to move some stuff here
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RestoreOperatorBoundaryData(Operator *op, RDyBoundary boundary, OperatorBoundaryData *boundary_data) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCheck(op == boundary_data->op, comm, PETSC_ERR_USER, "Could not restore operator boundary data: wrong operator");

  PetscCheck(boundary.index == boundary_data->boundary.index, comm, PETSC_ERR_USER, "Could not restore operator boundary data: wrong boundary");
  if (CeedEnabled()) {
    if (boundary_data->values.updated) {
      PetscCallCEED(CeedVectorRestoreArray(boundary_data->values.ceed.vec, &boundary_data->values.ceed.data));
    }
    if (boundary_data->fluxes.updated) {
      PetscCallCEED(CeedVectorRestoreArray(boundary_data->fluxes.ceed.vec, &boundary_data->fluxes.ceed.data));
      PetscCallCEED(CeedVectorSetValue(boundary_data->fluxes.ceed.vec, 0.0));  // reset flux accumulation
    }
  } else {
    /* FIXME: soon we can have this vvv, but for now we're SWE-specific
       if (boundary_data->storage.updated) {
       PetscCall(VecRestoreArray(boundary_data->values.petsc.vec, &boundary_data->values.petsc.data));
       }
     */
    RiemannDataSWE bdata;
    PetscCall(GetPetscSWEDirichletBoundaryValues(op->petsc.context, boundary.index, &bdata));

    PetscRiemannDataSWE *data_swe = op->petsc.context;
    PetscReal            tiny_h   = data_swe->tiny_h;

    // set velocities from momenta
    for (PetscInt e = 0; e < boundary.num_edges; ++e) {
      if (bdata.h[e] > tiny_h) {
        bdata.u[e] = bdata.hu[e] / bdata.h[e];
        bdata.v[e] = bdata.hv[e] / bdata.h[e];
      } else {
        bdata.u[e] = 0.0;
        bdata.v[e] = 0.0;
      }
    }
  }

  op->lock.boundary_data[boundary_data->boundary.index] = NULL;
  *boundary_data                                        = (OperatorBoundaryData){0};
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GetOperatorSourceData(Operator *op, OperatorSourceData *source_data) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));

  *source_data = (OperatorSourceData){0};
  PetscCheck(!op->lock.source_data, comm, PETSC_ERR_USER, "Could not acquire lock on source data -- another entity has access");
  op->lock.source_data        = source_data;
  source_data->op             = op;
  source_data->num_components = op->num_components;

  if (CeedEnabled()) {
    // NOTE: our SWE-specific source operator has only one sub operator
    CeedOperator *sub_ops;
    PetscCallCEED(CeedCompositeOperatorGetSubList(op->ceed.source_operator, &sub_ops));
    CeedOperator source_op = sub_ops[0];

    // fetch the relevant vector
    CeedOperatorField field;
    PetscCallCEED(CeedOperatorGetFieldByName(source_op, "swe_src", &field));  // FIXME: only valid for SWE
    PetscCallCEED(CeedOperatorFieldGetVector(field, &source_data->sources.ceed.vec));
  } else {
    source_data->sources.petsc.vec = op->petsc.sources;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// sets values of the source term for the given component
PetscErrorCode SetOperatorSourceValues(OperatorSourceData *source_data, PetscInt component, PetscReal *source_values) {
  PetscFunctionBegin;

  if (CeedEnabled()) {
    // if this is the first update, get access to the vector's data
    if (!source_data->sources.updated) {
      PetscCallCEED(CeedVectorGetArray(source_data->sources.ceed.vec, CEED_MEM_HOST, &source_data->sources.ceed.data));
      source_data->sources.updated = PETSC_TRUE;
    }

    // reshape for multicomponent access
    CeedScalar(*values)[source_data->num_components];
    *((CeedScalar **)&values) = source_data->sources.ceed.data;

    // set the values
    for (CeedInt i = 0; i < source_data->op->mesh->num_owned_cells; ++i) {
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
    for (PetscInt i = 0; i < source_data->op->mesh->num_owned_cells; ++i) {
      s[i * source_data->num_components + component] = source_values[i];
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RestoreOperatorSourceData(Operator *op, OperatorSourceData *source_data) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCheck(op == source_data->op, comm, PETSC_ERR_USER, "Could not restore operator source data: wrong operator");

  if (CeedEnabled()) {
    if (source_data->sources.updated) {
      PetscCallCEED(CeedVectorRestoreArray(source_data->sources.ceed.vec, &source_data->sources.ceed.data));
    }
  } else {  // petsc
    if (source_data->sources.updated) {
      PetscCall(VecRestoreArray(source_data->sources.petsc.vec, &source_data->sources.petsc.data));
    }
  }
  source_data->op->lock.source_data = NULL;
  *source_data                      = (OperatorSourceData){0};
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GetOperatorMaterialData(Operator *op, OperatorMaterialData *material_data) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCheck(!op->lock.material_data, comm, PETSC_ERR_USER, "Could not acquire lock on material data -- another entity has access");

  *material_data         = (OperatorMaterialData){0};
  op->lock.material_data = material_data;
  material_data->op      = op;

  if (CeedEnabled()) {
    // NOTE: our SWE-specific source operator has only one sub operator
    CeedOperator *sub_ops;
    PetscCallCEED(CeedCompositeOperatorGetSubList(op->ceed.source_operator, &sub_ops));
    CeedOperator source_op = sub_ops[0];

    // fetch the relevant material property vectors
    CeedOperatorField field;
    PetscCallCEED(CeedOperatorGetFieldByName(source_op, "mannings_s", &field));  // FIXME: only valid for SWE
    PetscCallCEED(CeedOperatorFieldGetVector(field, &material_data->mannings.ceed.vec));
  } else {
    material_data->mannings.petsc.vec = op->petsc.sources;  // FIXME: incorrect!
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
    for (CeedInt i = 0; i < material_data->op->mesh->num_owned_cells; ++i) {
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
    for (PetscInt i = 0; i < material_data->op->mesh->num_owned_cells; ++i) {
      m[i] = material_values[i];
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RestoreOperatorMaterialData(Operator *op, OperatorMaterialData *material_data) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCheck(op == material_data->op, comm, PETSC_ERR_USER, "Could not restore operator material data: wrong operator");

  if (CeedEnabled()) {
    if (material_data->mannings.updated) {
      PetscCallCEED(CeedVectorRestoreArray(material_data->mannings.ceed.vec, &material_data->mannings.ceed.data));
    }
  } else {  // petsc
    if (material_data->mannings.updated) {
      PetscCall(VecRestoreArray(material_data->mannings.petsc.vec, &material_data->mannings.petsc.data));
    }
  }
  material_data->op->lock.material_data = NULL;
  *material_data                        = (OperatorMaterialData){0};
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GetOperatorFluxDivergenceData(Operator *op, OperatorFluxDivergenceData *flux_div_data) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCheck(!op->lock.flux_div_data, comm, PETSC_ERR_USER, "Could not acquire lock on flux divergence data -- another entity has access");

  *flux_div_data                = (OperatorFluxDivergenceData){0};
  op->lock.flux_div_data        = flux_div_data;
  flux_div_data->op             = op;
  flux_div_data->num_components = op->num_components;

  if (CeedEnabled()) {
    // NOTE: our SWE-specific source operator has only one sub operator
    CeedOperator *sub_ops;
    PetscCallCEED(CeedCompositeOperatorGetSubList(op->ceed.source_operator, &sub_ops));
    CeedOperator source_op = sub_ops[0];

    // fetch the relevant vector
    CeedOperatorField field;
    PetscCallCEED(CeedOperatorGetFieldByName(source_op, "riemannf", &field));
    PetscCallCEED(CeedOperatorFieldGetVector(field, &flux_div_data->storage.ceed.vec));
  } else {
    // FIXME: our PETSc implementation currently stores flux divergences in the RHS vector
    // flux_div_data->storage.petsc.vec = rdy->petsc.flux_divergences;
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
    for (CeedInt i = 0; i < flux_div_data->op->mesh->num_owned_cells; ++i) {
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
    for (PetscInt i = 0; i < flux_div_data->op->mesh->num_owned_cells; ++i) {
      div_f[i * flux_div_data->num_components + component] = flux_div_values[i];
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RestoreOperatorFluxDivergenceData(Operator *op, OperatorFluxDivergenceData *flux_div_data) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCheck(op == flux_div_data->op, comm, PETSC_ERR_USER, "Could not restore operator flux divergence data: wrong operator");

  if (CeedEnabled()) {
    if (flux_div_data->storage.updated) {
      PetscCallCEED(CeedVectorRestoreArray(flux_div_data->storage.ceed.vec, &flux_div_data->storage.ceed.data));
    }
  } else {  // petsc
    if (flux_div_data->storage.updated) {
      PetscCall(VecRestoreArray(flux_div_data->storage.petsc.vec, &flux_div_data->storage.petsc.data));
    }
  }
  flux_div_data->op->lock.flux_div_data = NULL;
  *flux_div_data                        = (OperatorFluxDivergenceData){0};
  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
