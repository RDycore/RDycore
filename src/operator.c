#include <ceed/ceed.h>
#include <petscdmceed.h>
#include <private/rdycoreimpl.h>
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

//-------------------------------------
// Operator Construction / Destruction
//-------------------------------------

// defines the computational domain for the operator
static PetscErrorCode SetOperatorDomain(Operator *op, DM dm, RDyMesh *mesh) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCheck(mesh, comm, PETSC_ERR_USER, "Operator mesh must be non-NULL");

  op->dm   = dm;
  op->mesh = mesh;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// creates a sequential vector appropriate for our chosen architecture
static PetscErrorCode CreateSequentialVector(MPI_Comm comm, PetscInt size, Vec *vec) {
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
  PetscCall(VecSetUp(*vec));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// defines the distinct regions of owned cells for the operator
static PetscErrorCode SetOperatorRegions(Operator *op, PetscInt num_regions, RDyRegion *regions) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCheck(num_regions > 0, comm, PETSC_ERR_USER, "Number of operator regions must be positive");
  PetscCheck(regions, comm, PETSC_ERR_USER, "Operator region array must be non-NULL");

  // the operator does not manage its boundaries or boundary conditions -- these
  // are managed by RDy itself
  op->num_regions = num_regions;
  op->regions     = regions;

  // allocate sequential vectors for each region
  if (!CeedEnabled()) {
    PetscCall(CreateSequentialVector(comm, op->num_components * op->mesh->num_owned_cells, &op->petsc.external_sources));

    PetscCall(PetscCalloc1(OPERATOR_NUM_MATERIAL_PROPERTIES, &op->petsc.material_properties));  // NOLINT(bugprone-sizeof-expression)
    for (PetscInt p = 0; p < OPERATOR_NUM_MATERIAL_PROPERTIES; ++p) {
      PetscCall(CreateSequentialVector(comm, op->num_components * op->mesh->num_owned_cells, &op->petsc.material_properties[p]));
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// defines the distinct boundaries (edges) for the operator
static PetscErrorCode SetOperatorBoundaries(Operator *op, PetscInt num_boundaries, RDyBoundary *boundaries, RDyCondition *conditions) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCheck(num_boundaries > 0, comm, PETSC_ERR_USER, "Number of operator boundaries must be positive");
  PetscCheck(boundaries, comm, PETSC_ERR_USER, "Operator boundary array must be non-NULL");
  PetscCheck(conditions, comm, PETSC_ERR_USER, "Operator boundary conditions array must be non-NULL");

  // the operator does not manage its boundaries or boundary conditions -- these
  // are managed by RDy itself
  op->num_boundaries      = num_boundaries;
  op->boundaries          = boundaries;
  op->boundary_conditions = conditions;

  if (!CeedEnabled()) {
    PetscCall(PetscCalloc1(num_boundaries, &op->petsc.boundary_values));  // NOLINT(bugprone-sizeof-expression)
    PetscCall(PetscCalloc1(num_boundaries, &op->petsc.boundary_fluxes));  // NOLINT(bugprone-sizeof-expression)

    for (PetscInt b = 0; b < num_boundaries; ++b) {
      PetscInt num_edges = boundaries[b].num_edges;
      PetscCall(CreateSequentialVector(comm, op->num_components * num_edges, &op->petsc.boundary_values[b]));
      PetscCall(CreateSequentialVector(comm, op->num_components * num_edges, &op->petsc.boundary_fluxes[b]));
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AddCeedSuboperators(Operator *op) {
  PetscFunctionBegin;

  // set up operators for the shallow water equations

  Ceed      ceed   = CeedContext();
  PetscReal tiny_h = op->config->physics.flow.tiny_h;

  //---------------
  // Flux Operator
  //---------------

  PetscCallCEED(CeedCompositeOperatorCreate(ceed, &op->ceed.flux));

  // suboperator 0: fluxes between interior cells
  CeedOperator interior_flux_op;
  PetscCall(CreateSWECeedInteriorFluxOperator(op->mesh, tiny_h, &interior_flux_op));
  PetscCallCEED(CeedCompositeOperatorAddSub(op->ceed.flux, interior_flux_op));
  PetscCallCEED(CeedOperatorDestroy(&interior_flux_op));

  // suboperators 1 to num_boundaries: fluxes on boundary edges
  for (CeedInt b = 0; b < op->num_boundaries; ++b) {
    CeedOperator boundary_flux_op;
    RDyBoundary  boundary  = op->boundaries[b];
    RDyCondition condition = op->boundary_conditions[b];
    PetscCall(CreateSWECeedBoundaryFluxOperator(op->mesh, boundary, condition, tiny_h, &boundary_flux_op));
    PetscCallCEED(CeedCompositeOperatorAddSub(op->ceed.flux, boundary_flux_op));
    PetscCallCEED(CeedOperatorDestroy(&boundary_flux_op));
  }

  if (0) PetscCallCEED(CeedOperatorView(op->ceed.flux, stdout));

  //-----------------
  // Source Operator
  //-----------------

  CeedOperator source_op;
  PetscCall(CreateSWECeedSourceOperator(op->mesh, tiny_h, &source_op));
  PetscCallCEED(CeedCompositeOperatorAddSub(op->ceed.source, source_op));
  PetscCallCEED(CeedOperatorDestroy(&source_op));

  if (0) PetscCallCEED(CeedOperatorView(op->ceed.source, stdout));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AddPetscSuboperators(Operator *op) {
  PetscFunctionBegin;

  // set up sub-operators for the shallow water equations
  // NOTE: unlike CEED operators, Petsc operators are not ref-counted, so we
  // NOTE: don't need to "destroy" them after adding them to composite operators

  PetscCall(PetscCompositeOperatorCreate(&op->petsc.flux));
  PetscCall(PetscCompositeOperatorCreate(&op->petsc.source));

  PetscReal tiny_h = op->config->physics.flow.tiny_h;

  // flux suboperator 0: fluxes between interior cells
  PetscOperator interior_flux_op;
  PetscCall(CreateSWEPetscInteriorFluxOperator(op->mesh, &op->diagnostics, tiny_h, &interior_flux_op));
  PetscCall(PetscCompositeOperatorAddSub(op->petsc.flux, interior_flux_op));

  // flux suboperators 1 to num_boundaries: fluxes on boundary edges
  for (CeedInt b = 0; b < op->num_boundaries; ++b) {
    PetscOperator boundary_flux_op;
    RDyBoundary   boundary  = op->boundaries[b];
    RDyCondition  condition = op->boundary_conditions[b];
    PetscCall(CreateSWEPetscBoundaryFluxOperator(op->mesh, boundary, condition, op->petsc.boundary_values[b], op->petsc.boundary_fluxes[b],
                                                 &op->diagnostics, tiny_h, &boundary_flux_op));
    PetscCall(PetscCompositeOperatorAddSub(op->petsc.flux, boundary_flux_op));
  }

  // domain-wide SWE source operator
  PetscOperator source_op;
  PetscCall(CreateSWEPetscSourceOperator(op->mesh, op->petsc.external_sources, op->petsc.material_properties[OPERATOR_MANNINGS], tiny_h, &source_op));
  PetscCall(PetscCompositeOperatorAddSub(op->petsc.source, source_op));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// sets up a field named "riemannf" for the source operator, associating
// it with a vector that stores flux divergences computed by the flux operator
static PetscErrorCode AddOperatorFluxDivergence(Operator *op) {
  PetscFunctionBegin;

  // Create a flux divergence PETSc vector with the same characteristics as
  // the solution vector
  PetscCall(DMCreateGlobalVector(op->dm, &op->flux_divergence));
  PetscCall(VecZeroEntries(op->flux_divergence));

  if (CeedEnabled()) {
    Ceed    ceed               = CeedContext();
    CeedInt num_comp           = op->num_components;
    CeedInt num_owned_cells    = op->mesh->num_owned_cells;
    CeedInt flux_div_strides[] = {num_comp, 1, num_comp};

    // create a vector that provides flux divergence data to the source operator
    CeedElemRestriction flux_div_restriction;
    PetscCallCEED(
        CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, num_comp, num_owned_cells * num_comp, flux_div_strides, &flux_div_restriction));
    PetscCallCEED(CeedElemRestrictionCreateVector(flux_div_restriction, &op->ceed.flux_divergence, NULL));

    // add this vector to all source sub-operators
    CeedInt num_source_suboperators;
    PetscCallCEED(CeedCompositeOperatorGetNumSub(op->ceed.source, &num_source_suboperators));
    CeedOperator *source_suboperators;
    PetscCallCEED(CeedCompositeOperatorGetSubList(op->ceed.source, &source_suboperators));
    for (CeedInt i = 0; i < num_source_suboperators; ++i) {
      PetscCallCEED(CeedOperatorSetField(source_suboperators[i], "riemannf", flux_div_restriction, CEED_BASIS_COLLOCATED, op->ceed.flux_divergence));
    }

    // clean up (the suboperators keep references to restrictions and vectors)
    PetscCallCEED(CeedElemRestrictionDestroy(&flux_div_restriction));
  } else {
    PetscCall(PetscOperatorSetField(op->petsc.source, "riemannf", op->flux_divergence));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// performs all work necessary to make the operator ready for use
static PetscErrorCode AddSuboperators(Operator *op) {
  PetscFunctionBegin;

  if (CeedEnabled()) {
    PetscCall(AddCeedSuboperators(op));
  } else {
    PetscCall(AddPetscSuboperators(op));
  }

  // set up our flux divergence vector(s)
  PetscCall(AddOperatorFluxDivergence(op));

  // initialize diagnostics
  PetscCall(ResetOperatorDiagnostics(op));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Creates an operator representing the system of equations described in the
/// given configuration.
/// @param [in]  config              the configuration defining the physics and numerics for the new operator
/// @param [in]  domain_dm           a DM representing the computational domain on which the operator is defined
/// @param [in]  domain_mesh         a mesh containing geometric and topological information for the domain
/// @param [in]  num_regions         the number of disjoint regions partitioning the computational domain
/// @param [in]  regions             an array of disjoint regions paratitioning the computational domain
/// @param [in]  num_boundaries      the number of distinct boundaries bounding the computational domain
/// @param [in]  boundaries          an array of distinct boundaries bounding the computational domain
/// @param [in]  boundary_conditions an array of boundary conditions corresponding to the domain boundaries
/// @param [out] op                  the newly created operator
PetscErrorCode CreateOperator(RDyConfig *config, DM domain_dm, RDyMesh *domain_mesh, PetscInt num_regions, RDyRegion *regions,
                              PetscInt num_boundaries, RDyBoundary *boundaries, RDyCondition *boundary_conditions, Operator **operator) {
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
  (*operator)->config         = config;
  (*operator)->num_components = 3;

  PetscCall(SetOperatorDomain(*operator, domain_dm, domain_mesh));
  PetscCall(SetOperatorBoundaries(*operator, num_boundaries, boundaries, boundary_conditions));
  PetscCall(SetOperatorRegions(*operator, num_regions, regions));
  PetscCall(AddSuboperators(*operator));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Frees all resources devoted to the operator.
/// @param [out] op the operator to be freed
PetscErrorCode DestroyOperator(Operator **op) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)(*op)->dm, &comm));

  PetscBool ceed_enabled = CeedEnabled();

  if (ceed_enabled) {
    PetscCallCEED(CeedOperatorDestroy(&((*op)->ceed.flux)));
    PetscCallCEED(CeedOperatorDestroy(&((*op)->ceed.source)));
    PetscCallCEED(CeedVectorDestroy(&((*op)->ceed.u_local)));
    PetscCallCEED(CeedVectorDestroy(&((*op)->ceed.rhs)));
    if ((*op)->ceed.flux_divergence) {
      PetscCallCEED(CeedVectorDestroy(&((*op)->ceed.flux_divergence)));
    }
  } else {  // petsc
    for (PetscInt b = 0; b < (*op)->num_boundaries; ++b) {
      PetscCall(VecDestroy(&(*op)->petsc.boundary_values[b]));
      PetscCall(VecDestroy(&(*op)->petsc.boundary_fluxes[b]));
    }
    PetscFree((*op)->petsc.boundary_values);
    PetscFree((*op)->petsc.boundary_fluxes);
    PetscCall(VecDestroy(&(*op)->petsc.external_sources));
    for (PetscInt p = 0; p < 1; ++p) {
      PetscCall(VecDestroy(&(*op)->petsc.material_properties[p]));
    }
    PetscFree((*op)->petsc.material_properties);
    PetscCall(PetscOperatorDestroy(&(*op)->petsc.flux));
    PetscCall(PetscOperatorDestroy(&(*op)->petsc.source));
  }
  if ((*op)->flux_divergence) {
    PetscCall(VecDestroy(&(*op)->flux_divergence));
  }

  PetscFree(*op);
  *op = NULL;

  PetscFunctionReturn(PETSC_SUCCESS);
}

//----------------------
// Operator Application
//----------------------

static inline CeedMemType MemTypeP2C(PetscMemType mem_type) { return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST; }

static PetscErrorCode ApplyCeedOperator(Operator *op, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBegin;

  // update the timestep for the ceed operators if necessary
  if (op->ceed.dt != dt) {
    op->ceed.dt = dt;

    CeedContextFieldLabel label;
    PetscCallCEED(CeedOperatorGetContextFieldLabel(op->ceed.flux, "time step", &label));
    PetscCallCEED(CeedOperatorSetContextDouble(op->ceed.flux, label, &op->ceed.dt));
    PetscCallCEED(CeedOperatorSetContextDouble(op->ceed.source, label, &op->ceed.dt));
  }

  //------------------
  // Flux Calculation
  //------------------

  {
    // point our CEED solution vector at our PETSc solution vector
    PetscMemType mem_type;
    PetscScalar *u_local_ptr;
    PetscCall(VecGetArrayAndMemType(u_local, &u_local_ptr, &mem_type));
    PetscCallCEED(CeedVectorSetArray(op->ceed.u_local, MemTypeP2C(mem_type), CEED_USE_POINTER, u_local_ptr));

    // point our CEED right-hand side vector at a PETSc right-hand side vector
    Vec f_local;
    PetscCall(DMGetLocalVector(op->dm, &f_local));
    PetscScalar *f_local_ptr;
    PetscCall(VecGetArrayAndMemType(f_local, &f_local_ptr, &mem_type));
    PetscCallCEED(CeedVectorSetArray(op->ceed.rhs, MemTypeP2C(mem_type), CEED_USE_POINTER, f_local_ptr));

    // apply the flux operator, computing flux divergences
    PetscCall(PetscLogEventBegin(RDY_CeedOperatorApply_, u_local, f_global, 0, 0));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCEED(CeedOperatorApply(op->ceed.flux, op->ceed.u_local, op->ceed.rhs, CEED_REQUEST_IMMEDIATE));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogEventEnd(RDY_CeedOperatorApply_, u_local, f_global, 0, 0));

    // accumulate f_local into f_global
    PetscCall(VecZeroEntries(f_global));
    PetscCall(DMLocalToGlobal(op->dm, f_local, ADD_VALUES, f_global));

    // reset our CeedVectors and restore our PETSc vectors
    PetscCallCEED(CeedVectorTakeArray(op->ceed.rhs, MemTypeP2C(mem_type), &f_local_ptr));
    PetscCallCEED(CeedVectorTakeArray(op->ceed.u_local, MemTypeP2C(mem_type), &u_local_ptr));

    PetscCall(VecRestoreArrayAndMemType(f_local, &f_local_ptr));
    PetscCall(VecRestoreArrayAndMemType(u_local, &u_local_ptr));

    PetscCall(DMRestoreLocalVector(op->dm, &f_local));
  }

  //--------------------
  // Source Calculation
  //--------------------

  {
    // point our CEED solution vector at our PETSc solution vector
    PetscMemType mem_type;
    PetscScalar *u_local_ptr;
    PetscCall(VecGetArrayAndMemType(u_local, &u_local_ptr, &mem_type));
    PetscCallCEED(CeedVectorSetArray(op->ceed.u_local, MemTypeP2C(mem_type), CEED_USE_POINTER, u_local_ptr));

    // copy the flux divergences out of f_global for use with the source operator
    PetscCall(VecCopy(f_global, op->flux_divergence));

    // point our flux divergence CeedVector at our flux divergence PETSc Vec
    PetscScalar *flux_div_ptr;
    PetscCall(VecGetArrayAndMemType(op->flux_divergence, &flux_div_ptr, &mem_type));
    PetscCallCEED(CeedVectorSetArray(op->ceed.flux_divergence, MemTypeP2C(mem_type), CEED_USE_POINTER, flux_div_ptr));

    // point our source CEED vector at f_global, where the final RHS ends up
    PetscScalar *f_global_ptr;
    PetscCall(VecGetArrayAndMemType(f_global, &f_global_ptr, &mem_type));
    PetscCallCEED(CeedVectorSetArray(op->ceed.sources, MemTypeP2C(mem_type), CEED_USE_POINTER, f_global_ptr));

    // apply the source operator(s)
    PetscCall(PetscLogEventBegin(RDY_CeedOperatorApply_, u_local, f_global, 0, 0));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCEED(CeedOperatorApply(op->ceed.source, op->ceed.u_local, op->ceed.sources, CEED_REQUEST_IMMEDIATE));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogEventEnd(RDY_CeedOperatorApply_, u_local, f_global, 0, 0));

    // reset our CeedVectors and restore our PETSc vectors
    PetscCallCEED(CeedVectorTakeArray(op->ceed.sources, MemTypeP2C(mem_type), &f_global_ptr));
    PetscCallCEED(CeedVectorTakeArray(op->ceed.flux_divergence, MemTypeP2C(mem_type), &flux_div_ptr));
    PetscCallCEED(CeedVectorTakeArray(op->ceed.u_local, MemTypeP2C(mem_type), &u_local_ptr));

    PetscCall(VecRestoreArrayAndMemType(f_global, &f_global_ptr));
    PetscCall(VecRestoreArrayAndMemType(op->flux_divergence, &flux_div_ptr));
    PetscCall(VecRestoreArrayAndMemType(u_local, &u_local_ptr));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ApplyPetscOperator(Operator *op, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBegin;

  // apply the composite PETSc flux operators
  PetscCall(PetscOperatorApply(op->petsc.flux, dt, u_local, f_global));

  // NOTE: we don't need a separate flux divergence vector currently, but if/when
  // NOTE: we do, we'd do this vvv
  //// copy the flux divergences out of f_global for use with the source operator
  // PetscCall(VecCopy(f_global, op->flux_divergence));

  // apply the composite PETSc source operators
  PetscCall(PetscOperatorApply(op->petsc.source, dt, u_local, f_global));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Applies the operator to a local solution vector, storing the result in the
/// given global vector.
/// @param [inout] op       the operator to be applied to the solution vector
/// @param [dt]    dt       the time step over which the operator is applied
/// @param [in]    u_local  the local solution vector to which the operator is applied
/// @param [inout] f_global the global vector storing the result of the application
PetscErrorCode ApplyOperator(Operator *op, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBegin;

  if (CeedEnabled()) {
    PetscCall(ApplyCeedOperator(op, dt, u_local, f_global));
  } else {
    PetscCall(ApplyPetscOperator(op, dt, u_local, f_global));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

//----------------------
// Operator Diagnostics
//----------------------

/// Resets all operator diagnostics so they can be re-accumulated.
/// @param [inout] op the operator for which diagnostics are reset
PetscErrorCode ResetOperatorDiagnostics(Operator *op) {
  PetscFunctionBegin;

  op->diagnostics.updated        = PETSC_FALSE;
  op->diagnostics.courant_number = (CourantNumberDiagnostics){
      .max_courant_num = 0.0,
      .global_edge_id  = -1,
      .global_cell_id  = -1,
  };

  PetscFunctionReturn(PETSC_SUCCESS);
}

// NOTE: the CEED Courant number logic below makes assumptions about the
// NOTE: structure of our composite operator that are valid for the shallow
// NOTE: water equations, but we might need to change it to suit more general
// NOTE: conditions

static PetscErrorCode CeedFindMaxCourantNumberInternalEdges(CeedOperator op_edges, RDyMesh *mesh, CourantNumberDiagnostics *courant_diags) {
  PetscFunctionBegin;

  // get the relevant interior sub-operator
  CeedOperator *sub_ops;
  PetscCallCEED(CeedCompositeOperatorGetSubList(op_edges, &sub_ops));
  CeedOperator interior_flux_op = sub_ops[0];

  // fetch the field
  CeedOperatorField courant_num;
  PetscCallCEED(CeedOperatorGetFieldByName(interior_flux_op, "courant_number", &courant_num));

  CeedVector courant_num_vec;
  PetscCallCEED(CeedOperatorFieldGetVector(courant_num, &courant_num_vec));

  CeedScalar(*courant_num_data)[2];  // values to the left/right of an edge
  PetscCallCEED(CeedVectorGetArray(courant_num_vec, CEED_MEM_HOST, (CeedScalar **)&courant_num_data));

  for (PetscInt ii = 0; ii < mesh->num_owned_internal_edges; ii++) {
    CeedScalar local_max           = fmax(courant_num_data[ii][0], courant_num_data[ii][1]);
    courant_diags->max_courant_num = fmax(courant_diags->max_courant_num, local_max);
  }
  PetscCallCEED(CeedVectorRestoreArray(courant_num_vec, (CeedScalar **)&courant_num_data));

  PetscFunctionReturn(PETSC_SUCCESS);
}
/// @brief Loops over all boundary conditions and finds the local maximum Courant number.
///        If needed, the data is moved from device to host.
/// @param [in] op_edges A CeedOperator object for edges
/// @param [in] num_boundaries Total number of boundaries
/// @param [in] boundaries A RDyBoundary object
/// @param [in] *max_courant_number Local maximum value of courant number
/// @return 0 on sucess, or a non-zero error code on failure
static PetscErrorCode CeedFindMaxCourantNumberBoundaryEdges(CeedOperator op_edges, PetscInt num_boundaries, RDyBoundary *boundaries,
                                                            CourantNumberDiagnostics *courant_diags) {
  PetscFunctionBegin;

  // loop over all boundaries
  for (PetscInt b = 0; b < num_boundaries; ++b) {
    RDyBoundary boundary = boundaries[b];

    // get the relevant boundary sub-operator
    CeedOperator *sub_ops;
    PetscCallCEED(CeedCompositeOperatorGetSubList(op_edges, &sub_ops));
    CeedOperator boundary_flux_op = sub_ops[1 + boundary.index];

    // fetch the field
    CeedOperatorField courant_num;
    PetscCallCEED(CeedOperatorGetFieldByName(boundary_flux_op, "courant_number", &courant_num));

    // get access to the data
    CeedVector courant_num_vec;
    PetscCallCEED(CeedOperatorFieldGetVector(courant_num, &courant_num_vec));
    CeedScalar(*courant_num_data)[1];
    PetscCallCEED(CeedVectorGetArray(courant_num_vec, CEED_MEM_HOST, (CeedScalar **)&courant_num_data));

    // find the maximum value
    for (PetscInt e = 0; e < boundary.num_edges; ++e) {
      courant_diags->max_courant_num = fmax(courant_diags->max_courant_num, courant_num_data[e][0]);
    }

    // restores the pointer
    PetscCallCEED(CeedVectorRestoreArray(courant_num_vec, (CeedScalar **)&courant_num_data));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Finds the global maximum Courant number across all internal and boundary edges.
/// @param [in] op_edges A CeedOperator object for edges
/// @param [in] num_boundaries Total number of boundaries
/// @param [in] boundaries A RDyBoundary object
/// @param [in] comm A MPI_Comm object
/// @param [out] *max_courant_number Global maximum value of courant number
/// @return 0 on sucess, or a non-zero error code on failure
static PetscErrorCode CeedFindMaxCourantNumber(CeedOperator op_edges, RDyMesh *mesh, PetscInt num_boundaries, RDyBoundary *boundaries, MPI_Comm comm,
                                               CourantNumberDiagnostics *courant_diags) {
  PetscFunctionBegin;

  PetscCall(CeedFindMaxCourantNumberInternalEdges(op_edges, mesh, courant_diags));
  PetscCall(CeedFindMaxCourantNumberBoundaryEdges(op_edges, num_boundaries, boundaries, courant_diags));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Finds the maximum Courant number for the libCEED and the PETSc version of SWE implementation
/// @param [inout] rdy An RDy object
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode UpdateCeedCourantNumberDiagnostics(Operator *op) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));

  CourantNumberDiagnostics *courant_num_diags = &op->diagnostics.courant_number;

  if (CeedEnabled()) {
    // we need to extract the maximum courant number from the operator in the
    // CEED case; in the PETSc case it's already set for this process
    PetscCall(CeedFindMaxCourantNumber(op->ceed.flux, op->mesh, op->num_boundaries, op->boundaries, comm, courant_num_diags));
  }

  // reduce the courant diagnostics across all processes
  MPI_Allreduce(MPI_IN_PLACE, courant_num_diags, 1, MPI_COURANT_NUMBER_DIAGNOSTICS, MPI_MAX_COURANT_NUMBER, comm);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Ensures that all operator diagnostics are updated. This can result in data
/// being copied between memory spaces.
/// @param [inout] op the operator for which diagnostics are updated
PetscErrorCode UpdateOperatorDiagnostics(Operator *op) {
  PetscFunctionBegin;
  if (!op->diagnostics.updated) {
    // our PETSc operators should update diagnostics in-place, so we only need
    // to update things for CEED
    if (CeedEnabled()) {
      PetscCall(UpdateCeedCourantNumberDiagnostics(op));
    }

    // reduce courant diagnostics across all processes
    MPI_Comm comm;
    PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
    MPI_Allreduce(MPI_IN_PLACE, &op->diagnostics.courant_number, 1, MPI_COURANT_NUMBER_DIAGNOSTICS, MPI_MAX_COURANT_NUMBER, comm);

    op->diagnostics.updated = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Retrieves diagnostics from an operator, whether or not they are updated.
/// @param [in]  op          the operator from which diagnostics are retrieved
/// @param [out] diagnostics the diagnostics retrieved from the operator
PetscErrorCode GetOperatorDiagnostics(Operator *op, OperatorDiagnostics *diagnostics) {
  PetscFunctionBegin;
  *diagnostics = op->diagnostics;
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
  PetscCallCEED(CeedCompositeOperatorGetSubList(op->ceed.flux, &sub_ops));
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
  PetscCallCEED(CeedCompositeOperatorGetSubList(op->ceed.flux, &sub_ops));
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

/// Provides read-write access to the operator's boundary value array data for
/// a given boundary.
/// @param [in]  op the operator for which data access is provided
/// @param [in]  boundary the boundary for which access to data is provided
/// @param [out] boundary_value_data the array data to which access is provided
PetscErrorCode GetOperatorBoundaryValues(Operator *op, RDyBoundary boundary, OperatorData *boundary_value_data) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCall(CheckOperatorBoundary(op, boundary, comm));

  boundary_value_data->num_components = op->num_components;
  PetscCall(PetscCalloc1(op->num_components, &boundary_value_data->values));
  if (CeedEnabled()) {
    PetscCall(GetCeedOperatorBoundaryData(op, boundary, "q_dirichlet", boundary_value_data));
  } else {  // petsc
    PetscCall(GetPetscOperatorBoundaryData(op, boundary, op->petsc.boundary_values[boundary.index], boundary_value_data));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RestoreOperatorBoundaryValues(Operator *op, RDyBoundary boundary, OperatorData *boundary_value_data) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCall(CheckOperatorBoundary(op, boundary, comm));

  if (CeedEnabled()) {
    PetscCallCEED(RestoreCeedOperatorBoundaryData(op, boundary, "q_dirichlet", boundary_value_data));
  } else {
    PetscCallCEED(RestorePetscOperatorBoundaryData(op, boundary, op->petsc.boundary_values[boundary.index], boundary_value_data));
  }
  PetscCall(PetscFree(boundary_value_data->values));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Provides read-write access to the operator's boundary flux array data for
/// a given boundary.
/// @param [in]  op the operator for which data access is provided
/// @param [in]  boundary the boundary for which access to flux data is provided
/// @param [out] boundary_flux_data the array data to which access is provided
PetscErrorCode GetOperatorBoundaryFluxes(Operator *op, RDyBoundary boundary, OperatorData *boundary_flux_data) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCall(CheckOperatorBoundary(op, boundary, comm));

  boundary_flux_data->num_components = op->num_components;
  PetscCall(PetscCalloc1(op->num_components, &boundary_flux_data->values));
  if (CeedEnabled()) {
    PetscCall(GetCeedOperatorBoundaryData(op, boundary, "flux", boundary_flux_data));
  } else {  // petsc
    PetscCall(GetPetscOperatorBoundaryData(op, boundary, op->petsc.boundary_fluxes[boundary.index], boundary_flux_data));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Releases access to the operator's boundary flux array data for a boundary
/// for which access was provided via @ref GetOperatorBoundaryFluxes. This
/// operation can cause data to be copied between memory spaces.
/// @param [in]  op the operator for which data access is released
/// @param [in]  boundary the boundary for which access to flux data is released
/// @param [out] boundary_flux_data the array data for which access is released
PetscErrorCode RestoreOperatorBoundaryFluxes(Operator *op, RDyBoundary boundary, OperatorData *boundary_flux_data) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCall(CheckOperatorBoundary(op, boundary, comm));

  if (CeedEnabled()) {
    PetscCallCEED(RestoreCeedOperatorBoundaryData(op, boundary, "flux", boundary_flux_data));
  } else {
    PetscCallCEED(RestorePetscOperatorBoundaryData(op, boundary, op->petsc.boundary_fluxes[boundary.index], boundary_flux_data));
  }
  PetscCall(PetscFree(boundary_flux_data->values));

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

static PetscErrorCode GetCeedSourceOperatorRegionData(Operator *op, RDyRegion region, const char *field_name, OperatorData *region_data) {
  PetscFunctionBegin;

  CeedOperator *sub_ops;
  PetscCallCEED(CeedCompositeOperatorGetSubList(op->ceed.source, &sub_ops));
  CeedOperator source_op = sub_ops[0];

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
    PetscCall(PetscCalloc1(region.num_owned_cells, &region_data->values[c]));
    for (PetscInt i = 0; i < region.num_owned_cells; ++i) {
      PetscInt owned_cell_id    = region.owned_cell_global_ids[i];
      region_data->values[c][i] = values[c][owned_cell_id];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RestoreCeedSourceOperatorRegionData(Operator *op, RDyRegion region, const char *field_name, OperatorData *region_data) {
  PetscFunctionBegin;

  CeedOperator *sub_ops;
  PetscCallCEED(CeedCompositeOperatorGetSubList(op->ceed.source, &sub_ops));
  CeedOperator source_op = sub_ops[1 + op->num_boundaries + region.index];

  // copy the data into place
  PetscInt num_components = op->num_components;
  CeedScalar(*values)[num_components];
  *((CeedScalar **)&values) = (CeedScalar *)region_data->array_pointer;
  for (PetscInt c = 0; c < num_components; ++c) {
    for (PetscInt i = 0; i < region.num_owned_cells; ++i) {
      PetscInt owned_cell_id   = region.owned_cell_global_ids[i];
      values[c][owned_cell_id] = region_data->values[c][i];
    }
    PetscCall(PetscFree(region_data->values[c]));
  }

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

static PetscErrorCode GetPetscSourceOperatorRegionData(Operator *op, RDyRegion region, Vec vec, OperatorData *region_data) {
  PetscFunctionBegin;

  PetscReal *data;
  PetscCall(VecGetArray(vec, &data));
  for (PetscInt c = 0; c < op->num_components; ++c) {
    PetscCall(PetscCalloc1(region.num_owned_cells, &region_data->values[c]));
    for (PetscInt ce = 0; ce < region.num_owned_cells; ++ce) {
      PetscInt owned_cell_id     = region.owned_cell_global_ids[ce];
      region_data->values[c][ce] = data[op->num_components * owned_cell_id + c];
    }
  }
  region_data->array_pointer = data;

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RestorePetscSourceOperatorRegionData(Operator *op, RDyRegion region, Vec vec, OperatorData *region_data) {
  PetscFunctionBegin;

  PetscReal *data = region_data->array_pointer;
  for (PetscInt c = 0; c < op->num_components; ++c) {
    for (PetscInt ce = 0; ce < region.num_owned_cells; ++ce) {
      PetscInt owned_cell_id                       = region.owned_cell_global_ids[ce];
      data[op->num_components * owned_cell_id + c] = region_data->values[c][ce];
    }
    PetscCall(PetscFree(region_data->values[c]));
  }
  PetscCall(VecRestoreArray(vec, &data));
  *region_data = (OperatorData){0};

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Provides read-write access to the operator's external source array data for
/// a given region.
/// @param [in]  op the operator for which data access is provided
/// @param [in]  region the region for which access to source data is provided
/// @param [out] source_data the array data to which access is provided
PetscErrorCode GetOperatorExternalSource(Operator *op, RDyRegion region, OperatorData *source_data) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCall(CheckOperatorRegion(op, region, comm));

  source_data->num_components = op->num_components;
  PetscCall(PetscCalloc1(op->num_components, &source_data->values));
  if (CeedEnabled()) {
    PetscCall(GetCeedSourceOperatorRegionData(op, region, "swe_src", source_data));
  } else {  // petsc
    PetscCall(GetPetscSourceOperatorRegionData(op, region, op->petsc.external_sources, source_data));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Releases access to the operator's external source array data for a region
/// for which access was provided via @ref GetOperatorExternalSource. This
/// operation can cause data to be copied between memory spaces.
/// @param [in]  op the operator for which data access is released
/// @param [in]  region the region for which access to source data is released
/// @param [out] source_data the array data for which access is released
PetscErrorCode RestoreOperatorExternalSource(Operator *op, RDyRegion region, OperatorData *source_data) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCall(CheckOperatorRegion(op, region, comm));

  if (CeedEnabled()) {
    PetscCallCEED(RestoreCeedSourceOperatorRegionData(op, region, "swe_src", source_data));
  } else {
    PetscCallCEED(RestorePetscSourceOperatorRegionData(op, region, op->petsc.external_sources, source_data));
  }
  PetscCall(PetscFree(source_data->values));

  PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------
// Regional Material Property Operator Data
//------------------------------------------

const char *material_property_names[] = {
    "manning",
};

/// Provides read-write access to the operator's material property array data
/// for a given region and material property.
/// @param [in]  op the operator for which data access is provided
/// @param [in]  region the region for which access to material propety data is provided
/// @param [in]  property_id the ID for the desired material property
/// @param [out] property_data the array data to which access is provided
PetscErrorCode GetOperatorMaterialProperty(Operator *op, RDyRegion region, OperatorMaterialPropertyId property_id, OperatorData *property_data) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCall(CheckOperatorRegion(op, region, comm));

  // NOTE: only single-component material property data currently supported!
  property_data->num_components = 1;
  PetscCall(PetscCalloc1(property_data->num_components, &property_data->values));

  switch (property_id) {
    case OPERATOR_MANNINGS:
      if (CeedEnabled()) {
        PetscCall(GetCeedSourceOperatorRegionData(op, region, material_property_names[property_id], property_data));
      } else {
        PetscCall(GetPetscSourceOperatorRegionData(op, region, op->petsc.material_properties[property_id], property_data));
      }
    default:
      PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "Invalid material property ID: %u", property_id);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Releases access to the operator's material property array data for a region
/// and property for which access was provided via @ref GetOperatorMaterialProperty.
/// This operation can cause data to be copied between memory spaces.
/// @param [in]  op the operator for which data access is released
/// @param [in]  region the region for which access to material propety data is released
/// @param [in]  property_id the ID for the desired material property
/// @param [out] property_data the array data for which access is released
PetscErrorCode RestoreOperatorMaterialProperty(Operator *op, RDyRegion region, OperatorMaterialPropertyId property_id, OperatorData *property_data) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCall(CheckOperatorRegion(op, region, comm));

  switch (property_id) {
    case OPERATOR_MANNINGS:
      if (CeedEnabled()) {
        PetscCall(RestoreCeedSourceOperatorRegionData(op, region, material_property_names[property_id], property_data));
      } else {
        PetscCall(RestorePetscSourceOperatorRegionData(op, region, op->petsc.material_properties[property_id], property_data));
      }
    default:
      PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "Invalid material property ID: %u", property_id);
  }
  PetscCall(PetscFree(property_data->values));

  PetscFunctionReturn(PETSC_SUCCESS);
}

//-----------------
// PETSc operators
//-----------------

/// Creates a new PetscOperator with behavior defined by the given arguments.
/// @param [in]  context a pointer to a data structure used by the operator implementation
/// @param [in]  apply   the function called by PetscOperatorApply
/// @param [in]  destroy the function called by PetscOperatorDestroy
/// @param [out] op      the PetscOperator created by this call
PetscErrorCode PetscOperatorCreate(void *context, PetscErrorCode (*apply)(void *, PetscOperatorFields, PetscReal, Vec, Vec),
                                   PetscErrorCode (*destroy)(void *), PetscOperator *op) {
  PetscFunctionBegin;
  PetscCall(PetscCalloc1(1, op));
  (*op)->fields       = (PetscOperatorFields){0};
  (*op)->is_composite = PETSC_FALSE;
  (*op)->context      = context;
  (*op)->apply        = apply;
  (*op)->destroy      = destroy;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Releases resources allocated to the given PetscOperator.
/// @param [inout] op the PetscOperator to be destroyed
PetscErrorCode PetscOperatorDestroy(PetscOperator *op) {
  PetscFunctionBegin;
  if ((*op)->fields.fields) {
    PetscFree((*op)->fields.fields);
  }
  PetscCall((*op)->destroy((*op)->context));
  PetscFree(*op);
  *op = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Applies the given PetscOperator to a local solution vector, storing the
/// results in a global "right-hand-side" vector.
/// @param [inout] op       the operator being applyed to the solution vector
/// @param [in]    dt       the timestep over which the operator is applied
/// @param [in]    u_local  the local solution vector to which the operator is applied
/// @param [inout] f_global the global vector storing the results of the application
PetscErrorCode PetscOperatorApply(PetscOperator op, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBegin;
  PetscCall(op->apply(op->context, op->fields, dt, u_local, f_global));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  PetscInt       num_suboperators, capacity;
  PetscOperator *suboperators;
} PetscCompositeOperator;

static PetscErrorCode PetscCompositeOperatorApply(void *context, PetscOperatorFields fields, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBegin;
  PetscCompositeOperator *composite = context;
  for (PetscInt i = 0; i < composite->num_suboperators; ++i) {
    PetscCall(PetscOperatorApply(composite->suboperators[i], dt, u_local, f_global));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscCompositeOperatorDestroy(void *context) {
  PetscFunctionBegin;
  PetscCompositeOperator *composite = context;
  for (PetscInt i = 0; i < composite->num_suboperators; ++i) {
    PetscCall(PetscOperatorDestroy(&composite->suboperators[i]));
  }
  PetscFree(composite->suboperators);
  PetscFree(composite);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Creates a PetscOperator that applies sub-operators in sequence.
/// @param [out] a pointer to the created (empty) composite operator
PetscErrorCode PetscCompositeOperatorCreate(PetscOperator *op) {
  PetscFunctionBegin;
  PetscCompositeOperator *composite;
  PetscCall(PetscCalloc1(1, &composite));
  *composite = (PetscCompositeOperator){0};
  PetscCall(PetscOperatorCreate(composite, PetscCompositeOperatorApply, PetscCompositeOperatorDestroy, op));
  (*op)->is_composite = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Appends a sub-operator to the given composite PetscOperator (which takes ownership).
/// @param [inout] op     the composite operator to which a sub-operator is appended
/// @param [in]    sub_op the suboperator to be appended to the composite operator
PetscErrorCode PetscCompositeOperatorAddSub(PetscOperator op, PetscOperator sub_op) {
  PetscFunctionBegin;
  PetscCompositeOperator *composite = op->context;
  if (composite->num_suboperators + 1 > composite->capacity) {
    composite->capacity = (composite->capacity > 0) ? 2 * composite->capacity : 8;
    PetscCall(PetscRealloc(sizeof(PetscOperator) * composite->capacity, &composite->suboperators));
  }
  composite->suboperators[composite->num_suboperators] = sub_op;
  ++composite->num_suboperators;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Retrieves a PETSc Vec corresponding to the PetscOperator field with the given name.
/// This is useful for extracting PETSc Vecs for use within operator apply() functions.
/// @param [in]  fields the set of available fields
/// @param [in]  name   the name of the desired field
/// @param [out] vec    points to the field with the desired name, or to NULL if no such field exists
PetscErrorCode PetscOperatorFieldsGet(PetscOperatorFields fields, const char *name, Vec *vec) {
  PetscFunctionBegin;
  for (PetscInt f = 0; f < fields.num_fields; ++f) {
    if (!strncmp(name, fields.fields[f].name, MAX_NAME_LEN)) {  // found it!
      *vec = fields.fields[f].vec;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  *vec = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AppendOperatorField(PetscOperatorFields *fields, PetscOperatorField field) {
  PetscFunctionBegin;

  if (fields->num_fields + 1 > fields->capacity) {
    fields->capacity = (fields->capacity > 0) ? 2 * fields->capacity : 8;
    PetscCall(PetscRealloc(sizeof(PetscOperatorField) * fields->capacity, &fields->fields));
  }
  fields->fields[fields->num_fields] = field;
  ++fields->num_fields;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Adds a field with the given name and Vec to the given PETSc operator. If
/// this is a composite operator, the field is set on all sub-operators. The
/// operator does NOT manage memory for the PETSc Vec.
/// @param [inout] op   the operator for which a field is set
/// @param [in]    name the name of the field
/// @param [in]    vec  the PETSc Vec storing the field's data (NOT managed by the operator)
PetscErrorCode PetscOperatorSetField(PetscOperator op, const char *name, Vec vec) {
  PetscFunctionBegin;
  // look for an existing field of this name
  PetscInt index = -1;
  if (op->fields.num_fields > 0) {
    for (PetscInt f = 0; f < op->fields.num_fields; ++f) {
      if (!strncmp(name, op->fields.fields[f].name, MAX_NAME_LEN)) {
        index = f;
      }
    }
  }

  // replace the existing field or append a new one
  PetscOperatorField field = {
      .vec = vec,
  };
  strncpy(field.name, name, MAX_NAME_LEN);
  if (index != -1) {
    op->fields.fields[index] = field;
  } else {
    PetscCall(AppendOperatorField(&op->fields, field));
  }

  if (op->is_composite) {
    PetscCompositeOperator *composite = op->context;
    for (PetscInt i = 0; i < composite->num_suboperators; ++i) {
      PetscCall(PetscOperatorSetField(composite->suboperators[i], name, vec));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
