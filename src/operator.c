#include <ceed/ceed.h>
#include <petscdmceed.h>
#include <private/rdycoreimpl.h>
#include <private/rdyoperatorimpl.h>
#include <private/rdysedimentimpl.h>
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

  // create bookkeeping vectors
  if (CeedEnabled()) {
    Ceed     ceed     = CeedContext();
    PetscInt num_comp = op->num_components;
    PetscCallCEED(CeedVectorCreate(ceed, mesh->num_cells * num_comp, &op->ceed.u_local));
    PetscCallCEED(CeedVectorCreate(ceed, mesh->num_cells * num_comp, &op->ceed.rhs));
    PetscCallCEED(CeedVectorCreate(ceed, mesh->num_owned_cells * num_comp, &op->ceed.sources));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// creates a sequential vector appropriate for our chosen architecture
static PetscErrorCode CreateSequentialVector(MPI_Comm comm, PetscInt size, PetscInt block_size, Vec *vec) {
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
  PetscCall(VecSetBlockSize(*vec, block_size));
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

  // allocate sequential vectors that store sources and material properties for
  // the PETSc operator, similar to how restrictions are used in CEED
  if (!CeedEnabled()) {
    PetscCall(CreateSequentialVector(comm, op->num_components * op->mesh->num_owned_cells, op->num_components, &op->petsc.external_sources));
    PetscCall(CreateSequentialVector(comm, op->mesh->num_owned_cells, NUM_MATERIAL_PROPERTIES, &op->petsc.material_properties));
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
    // in the PETSc case, allocate sequential vectors for storing boundary values
    // and fluxes, similar to how restrictions are used in CEED
    PetscCall(PetscCalloc1(num_boundaries, &op->petsc.boundary_values));        // NOLINT(bugprone-sizeof-expression)
    PetscCall(PetscCalloc1(num_boundaries, &op->petsc.boundary_fluxes));        // NOLINT(bugprone-sizeof-expression)
    PetscCall(PetscCalloc1(num_boundaries, &op->petsc.boundary_fluxes_accum));  // NOLINT(bugprone-sizeof-expression)

    for (PetscInt b = 0; b < num_boundaries; ++b) {
      PetscInt num_edges = boundaries[b].num_edges;
      PetscCall(CreateSequentialVector(comm, op->num_components * num_edges, op->num_components, &op->petsc.boundary_values[b]));
      PetscCall(CreateSequentialVector(comm, op->num_components * num_edges, op->num_components, &op->petsc.boundary_fluxes[b]));
      PetscCall(CreateSequentialVector(comm, op->num_components * num_edges, op->num_components, &op->petsc.boundary_fluxes_accum[b]));
    }
  }

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
    PetscCallCEED(CeedOperatorCompositeGetNumSub(op->ceed.source, &num_source_suboperators));
    CeedOperator *source_suboperators;
    PetscCallCEED(CeedOperatorCompositeGetSubList(op->ceed.source, &source_suboperators));
    for (CeedInt i = 0; i < num_source_suboperators; ++i) {
      PetscCallCEED(CeedOperatorSetField(source_suboperators[i], "riemannf", flux_div_restriction, CEED_BASIS_NONE, op->ceed.flux_divergence));
    }

    // clean up (the suboperators keep references to restrictions and vectors)
    PetscCallCEED(CeedElemRestrictionDestroy(&flux_div_restriction));
  } else {
    PetscCall(PetscOperatorSetField(op->petsc.source, "riemannf", op->flux_divergence));
  }

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
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode CreateOperator(RDyConfig *config, DM domain_dm, RDyMesh *domain_mesh, PetscInt num_comp, PetscInt num_regions, RDyRegion *regions,
                              PetscInt num_boundaries, RDyBoundary *boundaries, RDyCondition *boundary_conditions, Operator **operator) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)domain_dm, &comm));

  // check our arguments
  PetscCheck(domain_mesh, comm, PETSC_ERR_USER, "Cannot create an operator with no mesh");
  PetscCheck(num_regions > 0, comm, PETSC_ERR_USER, "Cannot create an operator with no regions");
  // NOTE: num_boundaries can be zero in a subdomain in a parallel simulation

  PetscCall(PetscCalloc1(1, operator));
  (*operator)->config         = config;
  (*operator)->num_components = num_comp;

  PetscCall(SetOperatorDomain(*operator, domain_dm, domain_mesh));
  if (num_boundaries > 0) {
    // set up boundaries for the operator, allocating any necessary storage
    // (e.g. sequential vectors for PETSc operator)
    PetscCall(SetOperatorBoundaries(*operator, num_boundaries, boundaries, boundary_conditions));
  }

  // set up regions for the operator, allocating any necessary storage
  // (e.g. sequential vectors for PETSc operator)
  PetscCall(SetOperatorRegions(*operator, num_regions, regions));

  // construct CEED or PETSc versions of the flux/sources operators based on
  // our configuration
  if (CeedEnabled()) {
    // register a logging event for applying our CEED operator
    static PetscBool first_time = PETSC_TRUE;
    if (first_time) {
      PetscCall(PetscLogEventRegister("CeedOperatorApp", RDY_CLASSID, &RDY_CeedOperatorApply_));
      first_time = PETSC_FALSE;
    }

    PetscCall(CreateCeedFluxOperator((*operator)->config, (*operator)->mesh, (*operator)->num_boundaries, (*operator)->boundaries,
                                     (*operator)->boundary_conditions, &(*operator)->ceed.flux));
    PetscCall(CreateCeedSourceOperator((*operator)->config, (*operator)->mesh, &(*operator)->ceed.source));
  } else {
    PetscCall(CreatePetscFluxOperator((*operator)->config, (*operator)->mesh, (*operator)->num_boundaries, (*operator)->boundaries,
                                      (*operator)->boundary_conditions, (*operator)->petsc.boundary_values, (*operator)->petsc.boundary_fluxes,
                                      (*operator)->petsc.boundary_fluxes_accum, &(*operator)->diagnostics, &(*operator)->petsc.flux));
    PetscCall(CreatePetscSourceOperator((*operator)->config, (*operator)->mesh, (*operator)->petsc.external_sources,
                                        (*operator)->petsc.material_properties, &(*operator)->petsc.source));
  }

  // set up our flux divergence vector(s)
  PetscCall(AddOperatorFluxDivergence(*operator));

  // initialize diagnostics
  PetscCall(ResetOperatorDiagnostics(*operator));

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
    PetscCallCEED(CeedVectorDestroy(&((*op)->ceed.sources)));
    if ((*op)->ceed.flux_divergence) {
      PetscCallCEED(CeedVectorDestroy(&((*op)->ceed.flux_divergence)));
    }
  } else {  // petsc
    for (PetscInt b = 0; b < (*op)->num_boundaries; ++b) {
      PetscCall(VecDestroy(&(*op)->petsc.boundary_values[b]));
      PetscCall(VecDestroy(&(*op)->petsc.boundary_fluxes[b]));
      PetscCall(VecDestroy(&(*op)->petsc.boundary_fluxes_accum[b]));
    }
    PetscFree((*op)->petsc.boundary_values);
    PetscFree((*op)->petsc.boundary_fluxes);
    PetscFree((*op)->petsc.boundary_fluxes_accum);
    PetscCall(VecDestroy(&(*op)->petsc.external_sources));
    PetscCall(VecDestroy(&(*op)->petsc.material_properties));
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
    PetscCallCEED(CeedOperatorGetContextFieldLabel(op->ceed.source, "time step", &label));
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
    PetscCall(VecZeroEntries(f_local));
    PetscScalar *f_local_ptr;
    PetscCall(VecGetArrayAndMemType(f_local, &f_local_ptr, &mem_type));
    PetscCallCEED(CeedVectorSetArray(op->ceed.rhs, MemTypeP2C(mem_type), CEED_USE_POINTER, f_local_ptr));

    // apply the flux operator, computing flux divergences
    PetscCall(PetscLogEventBegin(RDY_CeedOperatorApply_, u_local, f_global, 0, 0));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCEED(CeedOperatorApplyAdd(op->ceed.flux, op->ceed.u_local, op->ceed.rhs, CEED_REQUEST_IMMEDIATE));
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

/// @brief The two PETSc operators (flux and source) are applied to fill up the global
///        right hand side of the discretized equation
/// @param [in] op         an Operator struct
/// @param [in] dt         the time step
/// @param [in] u_local    the solution Vec containing locally-owned and ghost cells
/// @param [out] f_global  the global right hand side Vec to be evaluated at time t
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode ApplyPetscOperator(Operator *op, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBegin;

  // apply the composite PETSc flux operators
  PetscCall(PetscOperatorApply(op->petsc.flux, dt, u_local, f_global));

  // copy the flux divergences out of f_global for use with the source operator
  PetscCall(VecCopy(f_global, op->flux_divergence));

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

// MPI datatype corresponding to CourantNumberDiagnostics. Created during
// CreateSWEOperator.
MPI_Datatype MPI_COURANT_NUMBER_DIAGNOSTICS = MPI_DATATYPE_NULL;

// MPI operator used to determine the prevailing diagnostics for the maximum
// courant number on all processes. Created during CreateSWEOperator.
MPI_Op MPI_MAX_COURANT_NUMBER = MPI_OP_NULL;

// function implementing the above MPI operator
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

// this function destroys the above MPI datatype and operator
static void DestroyCourantNumberDiagnostics(void) {
  MPI_Op_free(&MPI_MAX_COURANT_NUMBER);
  MPI_Type_free(&MPI_COURANT_NUMBER_DIAGNOSTICS);
}

// this function initializes some MPI machinery for the above Courant number
// diagnostics, and is called by CreateBasicOperator, which is called when a
// CEED or PETSc Operator is created
PetscErrorCode InitCourantNumberDiagnostics(void) {
  PetscFunctionBegin;

  static PetscBool initialized = PETSC_FALSE;

  if (!initialized) {
    // create an MPI data type for the CourantNumberDiagnostics struct
    const int      num_blocks             = 3;
    const int      block_lengths[3]       = {1, 1, 1};
    const MPI_Aint block_displacements[3] = {
        offsetof(CourantNumberDiagnostics, max_courant_num),
        offsetof(CourantNumberDiagnostics, global_edge_id),
        offsetof(CourantNumberDiagnostics, global_cell_id),
    };
    MPI_Datatype block_types[3] = {MPIU_REAL, MPI_INT, MPI_INT};
    MPI_Type_create_struct(num_blocks, block_lengths, block_displacements, block_types, &MPI_COURANT_NUMBER_DIAGNOSTICS);
    MPI_Type_commit(&MPI_COURANT_NUMBER_DIAGNOSTICS);

    // create a corresponding reduction operator for the new type
    MPI_Op_create(FindCourantNumberDiagnostics, 1, &MPI_MAX_COURANT_NUMBER);

    // make sure the operator and the type are destroyed upon exit
    PetscCall(RDyOnFinalize(DestroyCourantNumberDiagnostics));

    initialized = PETSC_TRUE;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
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
  PetscCallCEED(CeedOperatorCompositeGetSubList(op_edges, &sub_ops));
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
    PetscCallCEED(CeedOperatorCompositeGetSubList(op_edges, &sub_ops));
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

static PetscErrorCode CreateOperatorBoundaryData(Operator *op, RDyBoundary boundary, OperatorData *data) {
  PetscFunctionBegin;
  data->num_components = op->num_components;
  PetscCall(PetscCalloc1(op->num_components, &data->values));
  for (PetscInt c = 0; c < op->num_components; ++c) {
    PetscCall(PetscCalloc1(boundary.num_edges, &data->values[c]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroyOperatorData(OperatorData *data) {
  PetscFunctionBegin;
  for (PetscInt c = 0; c < data->num_components; ++c) {
    PetscCall(PetscFree(data->values[c]));
  }
  PetscCall(PetscFree(data->values));
  *data = (OperatorData){0};
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GetCeedOperatorBoundaryData(Operator *op, RDyBoundary boundary, const char *field_name, OperatorData *boundary_data) {
  PetscFunctionBegin;

  // get the relevant boundary sub-operator
  CeedOperator *sub_ops;
  PetscCallCEED(CeedOperatorCompositeGetSubList(op->ceed.flux, &sub_ops));
  CeedOperator sub_op = sub_ops[1 + boundary.index];

  // fetch the relevant vector
  CeedOperatorField field;
  PetscCallCEED(CeedOperatorGetFieldByName(sub_op, field_name, &field));
  if (boundary.num_edges == 208) printf("CeedOperatorGetFieldByName: field_name = %s; %d\n", field_name, boundary.num_edges);
  CeedVector vec;
  PetscCallCEED(CeedOperatorFieldGetVector(field, &vec));

  // copy out operator data
  PetscInt num_comp = boundary_data->num_components;
  PetscCallCEED(CeedVectorGetArray(vec, CEED_MEM_HOST, &boundary_data->array_pointer));
  CeedScalar(*values)[num_comp];
  *((CeedScalar **)&values) = boundary_data->array_pointer;
  for (PetscInt c = 0; c < num_comp; ++c) {
    for (PetscInt e = 0; e < boundary.num_edges; ++e) {
      boundary_data->values[c][e] = values[e][c];
      if (e <= 10 && c == 0 && boundary.num_edges == 208) printf(" values[%d] = %18.16f\n", e, values[e][c]);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RestoreCeedOperatorBoundaryData(Operator *op, RDyBoundary boundary, const char *field_name, OperatorData *boundary_data) {
  PetscFunctionBegin;

  // get the relevant boundary sub-operator
  CeedOperator *sub_ops;
  PetscCallCEED(CeedOperatorCompositeGetSubList(op->ceed.flux, &sub_ops));
  CeedOperator sub_op = sub_ops[1 + boundary.index];

  // copy the data in
  PetscInt num_comp = boundary_data->num_components;
  CeedScalar(*values)[num_comp];
  *((CeedScalar **)&values) = boundary_data->array_pointer;
  for (PetscInt c = 0; c < num_comp; ++c) {
    for (PetscInt e = 0; e < boundary.num_edges; ++e) {
      values[e][c] = boundary_data->values[c][e];
    }
  }

  // release the array
  CeedOperatorField field;
  PetscCallCEED(CeedOperatorGetFieldByName(sub_op, field_name, &field));
  CeedVector vec;
  PetscCallCEED(CeedOperatorFieldGetVector(field, &vec));
  PetscCallCEED(CeedVectorRestoreArray(vec, &boundary_data->array_pointer));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GetPetscOperatorBoundaryData(Operator *op, RDyBoundary boundary, Vec vec, OperatorData *boundary_data) {
  PetscFunctionBegin;

  PetscReal *data;
  PetscCall(VecGetArray(vec, &data));
  PetscInt num_comp = boundary_data->num_components;
  for (PetscInt c = 0; c < num_comp; ++c) {
    for (PetscInt e = 0; e < boundary.num_edges; ++e) {
      boundary_data->values[c][e] = data[num_comp * e + c];
    }
  }
  boundary_data->array_pointer = data;

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RestorePetscOperatorBoundaryData(Operator *op, RDyBoundary boundary, Vec vec, OperatorData *boundary_data) {
  PetscFunctionBegin;

  PetscReal *data     = boundary_data->array_pointer;
  PetscInt   num_comp = boundary_data->num_components;
  for (PetscInt c = 0; c < num_comp; ++c) {
    for (PetscInt e = 0; e < boundary.num_edges; ++e) {
      data[num_comp * e + c] = boundary_data->values[c][e];
    }
  }
  PetscCall(VecRestoreArray(vec, &data));

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

  PetscCall(CreateOperatorBoundaryData(op, boundary, boundary_value_data));
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
  DestroyOperatorData(boundary_value_data);

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

  PetscCall(CreateOperatorBoundaryData(op, boundary, boundary_flux_data));
  boundary_flux_data->num_components = op->num_components;
  if (CeedEnabled()) {
    PetscCall(GetCeedOperatorBoundaryData(op, boundary, "flux_accumulated", boundary_flux_data));
  } else {  // petsc
    PetscCall(GetPetscOperatorBoundaryData(op, boundary, op->petsc.boundary_fluxes_accum[boundary.index], boundary_flux_data));
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
    PetscCallCEED(RestoreCeedOperatorBoundaryData(op, boundary, "flux_accumulated", boundary_flux_data));
  } else {
    PetscCallCEED(RestorePetscOperatorBoundaryData(op, boundary, op->petsc.boundary_fluxes_accum[boundary.index], boundary_flux_data));
  }
  DestroyOperatorData(boundary_flux_data);

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

static PetscErrorCode CreateOperatorRegionData(Operator *op, RDyRegion region, PetscInt num_comp, OperatorData *data) {
  PetscFunctionBegin;
  data->num_components = num_comp;
  PetscCall(PetscCalloc1(num_comp, &data->values));
  for (PetscInt c = 0; c < num_comp; ++c) {
    PetscCall(PetscCalloc1(region.num_owned_cells, &data->values[c]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GetCeedSourceOperatorRegionData(Operator *op, RDyRegion region, const char *field_name, OperatorData *region_data) {
  PetscFunctionBegin;

  CeedOperator *sub_ops;
  PetscCallCEED(CeedOperatorCompositeGetSubList(op->ceed.source, &sub_ops));
  CeedOperator source_op = sub_ops[0];

  // fetch the relevant vector
  CeedOperatorField field;
  PetscCallCEED(CeedOperatorGetFieldByName(source_op, field_name, &field));
  CeedVector vec;
  PetscCallCEED(CeedOperatorFieldGetVector(field, &vec));

  // copy out operator data
  PetscInt num_comp = region_data->num_components;
  PetscCallCEED(CeedVectorGetArray(vec, CEED_MEM_HOST, &region_data->array_pointer));
  CeedScalar(*values)[num_comp];
  *((CeedScalar **)&values) = region_data->array_pointer;
  for (PetscInt c = 0; c < num_comp; ++c) {
    for (PetscInt i = 0; i < region.num_owned_cells; ++i) {
      PetscInt owned_cell_id    = region.owned_cell_global_ids[i];
      region_data->values[c][i] = values[owned_cell_id][c];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RestoreCeedSourceOperatorRegionData(Operator *op, RDyRegion region, const char *field_name, OperatorData *region_data) {
  PetscFunctionBegin;

  CeedOperator *sub_ops;
  PetscCallCEED(CeedOperatorCompositeGetSubList(op->ceed.source, &sub_ops));
  CeedOperator source_op = sub_ops[0];

  // copy the data into place
  PetscInt num_comp = region_data->num_components;
  CeedScalar(*values)[num_comp];
  *((CeedScalar **)&values) = region_data->array_pointer;
  for (PetscInt c = 0; c < num_comp; ++c) {
    for (PetscInt i = 0; i < region.num_owned_cells; ++i) {
      PetscInt owned_cell_id   = region.owned_cell_global_ids[i];
      values[owned_cell_id][c] = region_data->values[c][i];
    }
  }

  // release the array
  CeedOperatorField field;
  PetscCallCEED(CeedOperatorGetFieldByName(source_op, field_name, &field));
  CeedVector vec;
  PetscCallCEED(CeedOperatorFieldGetVector(field, &vec));
  PetscCallCEED(CeedVectorRestoreArray(vec, &region_data->array_pointer));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GetPetscSourceOperatorRegionData(Operator *op, RDyRegion region, Vec vec, OperatorData *region_data) {
  PetscFunctionBegin;

  PetscReal *data;
  PetscCall(VecGetArray(vec, &data));
  PetscInt num_comp = region_data->num_components;
  for (PetscInt c = 0; c < num_comp; ++c) {
    for (PetscInt ce = 0; ce < region.num_owned_cells; ++ce) {
      PetscInt owned_cell_id     = region.owned_cell_global_ids[ce];
      region_data->values[c][ce] = data[num_comp * owned_cell_id + c];
    }
  }
  region_data->array_pointer = data;

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RestorePetscSourceOperatorRegionData(Operator *op, RDyRegion region, Vec vec, OperatorData *region_data) {
  PetscFunctionBegin;

  PetscInt   num_comp = region_data->num_components;
  PetscReal *data     = region_data->array_pointer;
  for (PetscInt c = 0; c < num_comp; ++c) {
    for (PetscInt ce = 0; ce < region.num_owned_cells; ++ce) {
      PetscInt owned_cell_id             = region.owned_cell_global_ids[ce];
      data[num_comp * owned_cell_id + c] = region_data->values[c][ce];
    }
  }
  PetscCall(VecRestoreArray(vec, &data));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Provides read-write access to the operator's external source array data for
/// a given region.
/// @param [in]  op the operator for which data access is provided
/// @param [in]  region the region for which access to source data is provided
/// @param [out] source_data the array data to which access is provided
PetscErrorCode GetOperatorRegionalExternalSource(Operator *op, RDyRegion region, OperatorData *source_data) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCall(CheckOperatorRegion(op, region, comm));

  PetscCall(CreateOperatorRegionData(op, region, op->num_components, source_data));
  if (CeedEnabled()) {
    PetscCall(GetCeedSourceOperatorRegionData(op, region, "ext_src", source_data));
  } else {  // petsc
    PetscCall(GetPetscSourceOperatorRegionData(op, region, op->petsc.external_sources, source_data));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Releases access to the operator's external source array data for a region
/// for which access was provided via @ref GetOperatorRegionalExternalSource. This
/// operation can cause data to be copied between memory spaces.
/// @param [in]  op the operator for which data access is released
/// @param [in]  region the region for which access to source data is released
/// @param [out] source_data the array data for which access is released
PetscErrorCode RestoreOperatorRegionalExternalSource(Operator *op, RDyRegion region, OperatorData *source_data) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCall(CheckOperatorRegion(op, region, comm));

  if (CeedEnabled()) {
    PetscCallCEED(RestoreCeedSourceOperatorRegionData(op, region, "ext_src", source_data));
  } else {
    PetscCallCEED(RestorePetscSourceOperatorRegionData(op, region, op->petsc.external_sources, source_data));
  }
  PetscCall(DestroyOperatorData(source_data));

  PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------
// Regional Material Property Operator Data
//------------------------------------------

/// Provides read-write access to the operator's material properties array data
/// for a given region.
/// @param [in]  op the operator for which data access is provided
/// @param [in]  region the region for which access to material propety data is provided
/// @param [out] property_data the array data to which access is provided
PetscErrorCode GetOperatorRegionalMaterialProperties(Operator *op, RDyRegion region, OperatorData *property_data) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCall(CheckOperatorRegion(op, region, comm));

  PetscCall(CreateOperatorRegionData(op, region, NUM_MATERIAL_PROPERTIES, property_data));
  if (CeedEnabled()) {
    PetscCall(GetCeedSourceOperatorRegionData(op, region, "mat_props", property_data));
  } else {
    PetscCall(GetPetscSourceOperatorRegionData(op, region, op->petsc.material_properties, property_data));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Releases access to the operator's regional material properties array data
/// for which access was provided via @ref GetOperatorRegionalMaterialProperty.
/// This operation can cause data to be copied between memory spaces.
/// @param [in]  op the operator for which data access is released
/// @param [in]  region the region for which access to material propety data is released
/// @param [out] property_data the array data for which access is released
PetscErrorCode RestoreOperatorRegionalMaterialProperties(Operator *op, RDyRegion region, OperatorData *property_data) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCall(CheckOperatorRegion(op, region, comm));

  if (CeedEnabled()) {
    PetscCall(RestoreCeedSourceOperatorRegionData(op, region, "mat_props", property_data));
  } else {
    PetscCall(RestorePetscSourceOperatorRegionData(op, region, op->petsc.material_properties, property_data));
  }
  PetscCall(DestroyOperatorData(property_data));

  PetscFunctionReturn(PETSC_SUCCESS);
}

//--------------------------------------
// Domain (Cell-Centered) Operator Data
//--------------------------------------

static PetscErrorCode CreateOperatorDomainData(Operator *op, PetscInt num_comp, OperatorData *data) {
  PetscFunctionBegin;
  data->num_components = num_comp;
  PetscCall(PetscCalloc1(num_comp, &data->values));
  for (PetscInt c = 0; c < num_comp; ++c) {
    PetscCall(PetscCalloc1(op->mesh->num_owned_cells, &data->values[c]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GetCeedSourceOperatorDomainData(Operator *op, const char *field_name, OperatorData *domain_data) {
  PetscFunctionBegin;

  CeedOperator *sub_ops;
  PetscCallCEED(CeedOperatorCompositeGetSubList(op->ceed.source, &sub_ops));
  CeedOperator source_op = sub_ops[0];

  // fetch the relevant vector
  CeedOperatorField field;
  PetscCallCEED(CeedOperatorGetFieldByName(source_op, field_name, &field));
  CeedVector vec;
  PetscCallCEED(CeedOperatorFieldGetVector(field, &vec));

  // copy out operator data
  PetscInt num_comp = domain_data->num_components;
  PetscCallCEED(CeedVectorGetArray(vec, CEED_MEM_HOST, &domain_data->array_pointer));
  CeedScalar(*values)[num_comp];
  *((CeedScalar **)&values) = domain_data->array_pointer;
  for (PetscInt c = 0; c < num_comp; ++c) {
    for (PetscInt i = 0; i < op->mesh->num_owned_cells; ++i) {
      domain_data->values[c][i] = values[i][c];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RestoreCeedSourceOperatorDomainData(Operator *op, const char *field_name, OperatorData *domain_data) {
  PetscFunctionBegin;

  CeedOperator *sub_ops;
  PetscCallCEED(CeedOperatorCompositeGetSubList(op->ceed.source, &sub_ops));
  CeedOperator source_op = sub_ops[0];

  // copy the data into place
  PetscInt num_comp = domain_data->num_components;
  CeedScalar(*values)[num_comp];
  *((CeedScalar **)&values) = domain_data->array_pointer;
  for (PetscInt c = 0; c < num_comp; ++c) {
    for (PetscInt i = 0; i < op->mesh->num_owned_cells; ++i) {
      values[i][c] = domain_data->values[c][i];
    }
  }

  // release the array
  CeedOperatorField field;
  PetscCallCEED(CeedOperatorGetFieldByName(source_op, field_name, &field));
  CeedVector vec;
  PetscCallCEED(CeedOperatorFieldGetVector(field, &vec));
  PetscCallCEED(CeedVectorRestoreArray(vec, &domain_data->array_pointer));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GetPetscSourceOperatorDomainData(Operator *op, Vec vec, OperatorData *domain_data) {
  PetscFunctionBegin;

  PetscReal *data;
  PetscCall(VecGetArray(vec, &data));
  PetscInt num_comp = domain_data->num_components;
  for (PetscInt c = 0; c < num_comp; ++c) {
    for (PetscInt i = 0; i < op->mesh->num_owned_cells; ++i) {
      domain_data->values[c][i] = data[num_comp * i + c];
    }
  }
  domain_data->array_pointer = data;

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RestorePetscSourceOperatorDomainData(Operator *op, Vec vec, OperatorData *domain_data) {
  PetscFunctionBegin;

  PetscInt   num_comp = domain_data->num_components;
  PetscReal *data     = domain_data->array_pointer;
  for (PetscInt c = 0; c < num_comp; ++c) {
    for (PetscInt i = 0; i < op->mesh->num_owned_cells; ++i) {
      data[num_comp * i + c] = domain_data->values[c][i];
    }
  }
  PetscCall(VecRestoreArray(vec, &data));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Provides read-write access to the operator's external source array data for
/// the entire domain.
/// @param [in]  op the operator for which data access is provided
/// @param [out] source_data the array data to which access is provided
PetscErrorCode GetOperatorDomainExternalSource(Operator *op, OperatorData *source_data) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));

  PetscCall(CreateOperatorDomainData(op, op->num_components, source_data));
  if (CeedEnabled()) {
    PetscCall(GetCeedSourceOperatorDomainData(op, "ext_src", source_data));
  } else {  // petsc
    PetscCall(GetPetscSourceOperatorDomainData(op, op->petsc.external_sources, source_data));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Releases access to the operator's external source array data for the entire
/// domain for which access was provided via @ref GetOperatorDomainExternalSource.
/// This operation can cause data to be copied between memory spaces.
/// @param [in]  op the operator for which data access is released
/// @param [out] source_data the array data for which access is released
PetscErrorCode RestoreOperatorDomainExternalSource(Operator *op, OperatorData *source_data) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));

  if (CeedEnabled()) {
    PetscCallCEED(RestoreCeedSourceOperatorDomainData(op, "ext_src", source_data));
  } else {
    PetscCallCEED(RestorePetscSourceOperatorDomainData(op, op->petsc.external_sources, source_data));
  }
  PetscCall(DestroyOperatorData(source_data));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Provides read-write access to the operator's material properties array data,
/// in which each component is a material property.
/// @param [in]  op the operator for which data access is provided
/// @param [out] property_data the array data to which access is provided
PetscErrorCode GetOperatorDomainMaterialProperties(Operator *op, OperatorData *property_data) {
  PetscFunctionBegin;

  PetscCall(CreateOperatorDomainData(op, NUM_MATERIAL_PROPERTIES, property_data));
  if (CeedEnabled()) {
    PetscCallCEED(GetCeedSourceOperatorDomainData(op, "mat_props", property_data));
  } else {
    PetscCall(GetPetscSourceOperatorDomainData(op, op->petsc.material_properties, property_data));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Releases access to the operator's material properties array data
/// for which access was provided via @ref GetOperatorDomainMaterialProperty.
/// This operation can cause data to be copied between memory spaces.
/// @param [in]  op the operator for which data access is released
/// @param [out] property_data the array data for which access is released
PetscErrorCode RestoreOperatorDomainMaterialProperties(Operator *op, OperatorData *property_data) {
  PetscFunctionBegin;

  if (CeedEnabled()) {
    PetscCall(RestoreCeedSourceOperatorDomainData(op, "mat_props", property_data));
  } else {
    PetscCall(RestorePetscSourceOperatorDomainData(op, op->petsc.material_properties, property_data));
  }
  PetscCall(DestroyOperatorData(property_data));

  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
