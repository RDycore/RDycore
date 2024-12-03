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
  PetscCall(PetscObjectGetComm((PetscObject)op->dm, &comm));
  PetscCheck(mesh, comm, PETSC_ERR_USER, "Operator mesh must be non-NULL");

  op->dm   = dm;
  op->mesh = mesh;

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

  // the operator does not manage its boundaries or boundary conditions -- these
  // are managed by RDy itself
  op->num_regions = num_regions;
  op->regions     = regions;

  // allocate sequential vectors for each region
  if (!CeedEnabled()) {
    PetscCall(PetscCalloc1(num_regions, &op->petsc.external_sources));  // NOLINT(bugprone-sizeof-expression)
    for (PetscInt r = 0; r < num_regions; ++r) {
      PetscCall(CreateSequentialVector(comm, op->num_components, regions[r].num_owned_cells, &op->petsc.external_sources[r]));
    }

    PetscCall(PetscCalloc1(OPERATOR_NUM_MATERIAL_PROPERTIES, &op->petsc.material_properties));
    for (PetscInt p = 0; p < OPERATOR_NUM_MATERIAL_PROPERTIES; ++p) {
      PetscCall(PetscCalloc1(num_regions, &op->petsc.material_properties[p]));  // NOLINT(bugprone-sizeof-expression)
      for (PetscInt r = 0; r < num_regions; ++r) {
        PetscCall(CreateSequentialVector(comm, op->num_components, regions[r].num_owned_cells, &op->petsc.material_properties[p][r]));
      }
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

  PetscReal tiny_h = op->config->physics.flow.tiny_h;

  // set up suboperators for the shallow water equations

  // suboperator 0: fluxes between interior cells
  CeedOperator interior_flux_op;
  PetscCall(CreateSWECeedInteriorFluxOperator(op->mesh, tiny_h, &interior_flux_op));
  PetscCallCEED(CeedCompositeOperatorAddSub(op->ceed.composite, interior_flux_op));
  PetscCallCEED(CeedOperatorDestroy(&interior_flux_op));

  // suboperators 1 to num_boundaries: fluxes on boundary edges
  for (CeedInt b = 0; b < op->num_boundaries; ++b) {
    CeedOperator boundary_flux_op;
    RDyBoundary  boundary  = op->boundaries[b];
    RDyCondition condition = op->boundary_conditions[b];
    PetscCall(CreateSWECeedBoundaryFluxOperator(op->mesh, boundary, condition, tiny_h, &boundary_flux_op));
    PetscCallCEED(CeedCompositeOperatorAddSub(op->ceed.composite, boundary_flux_op));
    PetscCallCEED(CeedOperatorDestroy(&boundary_flux_op));
  }

  // suboperators num_boundaries + 1 to num_boundaries + num_regions + 1: external sources
  for (CeedInt r = 0; r < op->num_regions; ++r) {
    CeedOperator source_op;
    RDyRegion    region = op->regions[r];
    PetscCall(CreateSWECeedSourceOperator(op->mesh, region, tiny_h, &source_op));
    PetscCallCEED(CeedCompositeOperatorAddSub(op->ceed.composite, source_op));
    PetscCallCEED(CeedOperatorDestroy(&source_op));
  }

  if (0) PetscCallCEED(CeedOperatorView(op->ceed.composite, stdout));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PreparePetscOperator(Operator *op) {
  PetscFunctionBegin;

  PetscCall(PetscCompositeOperatorCreate(&op->petsc.composite));

  PetscReal tiny_h = op->config->physics.flow.tiny_h;

  // set up suboperators for the shallow water equations

  // suboperator 0: fluxes between interior cells
  PetscOperator interior_flux_op;
  PetscCall(CreateSWEPetscInteriorFluxOperator(op->mesh, &op->diagnostics, tiny_h, &interior_flux_op));
  PetscCall(PetscCompositeOperatorAddSub(op->petsc.composite, interior_flux_op));
  PetscCall(PetscOperatorDestroy(&interior_flux_op));

  // suboperators 1 to num_boundaries: fluxes on boundary edges
  for (CeedInt b = 0; b < op->num_boundaries; ++b) {
    PetscOperator boundary_flux_op;
    RDyBoundary   boundary  = op->boundaries[b];
    RDyCondition  condition = op->boundary_conditions[b];
    PetscCall(CreateSWEPetscBoundaryFluxOperator(op->mesh, boundary, condition, op->petsc.boundary_values[b], op->petsc.boundary_fluxes[b],
                                                 &op->diagnostics, tiny_h, &boundary_flux_op));
    PetscCall(PetscCompositeOperatorAddSub(op->petsc.composite, boundary_flux_op));
    PetscCall(PetscOperatorDestroy(&boundary_flux_op));
  }

  // suboperators num_boundaries + 1 to num_boundaries + num_regions + 1: external sources
  for (CeedInt r = 0; r < op->num_regions; ++r) {
    PetscOperator source_op;
    RDyRegion     region = op->regions[r];
    PetscCall(CreateSWEPetscSourceOperator(op->mesh, region, op->petsc.external_sources[r], op->petsc.material_properties[OPERATOR_MANNINGS][r], tiny_h, &source_op));
    PetscCall(PetscCompositeOperatorAddSub(op->petsc.composite, source_op));
    PetscCall(PetscOperatorDestroy(&source_op));
  }

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
  (*operator)->config = config;

  PetscCall(SetOperatorDomain(*operator, domain_dm, domain_mesh));
  PetscCall(SetOperatorBoundaries(*operator, num_boundaries, boundaries, boundary_conditions));
  PetscCall(SetOperatorRegions(*operator, num_regions, regions));
  PetscCall(PrepareOperator(*operator));

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
    PetscCallCEED(CeedOperatorDestroy(&((*op)->ceed.composite)));
    PetscCallCEED(CeedVectorDestroy(&((*op)->ceed.u_local)));
    PetscCallCEED(CeedVectorDestroy(&((*op)->ceed.rhs)));
  } else {  // petsc
    for (PetscInt b = 0; b < (*op)->num_boundaries; ++b) {
      PetscCall(VecDestroy(&(*op)->petsc.boundary_values[b]));
      PetscCall(VecDestroy(&(*op)->petsc.boundary_fluxes[b]));
    }
    PetscFree((*op)->petsc.boundary_values);
    PetscFree((*op)->petsc.boundary_fluxes);
    for (PetscInt r = 0; r < (*op)->num_regions; ++r) {
      PetscCall(VecDestroy(&(*op)->petsc.external_sources[r]));
    }
    PetscFree((*op)->petsc.external_sources);
    for (PetscInt p = 0; p < 1; ++p) {
      for (PetscInt r = 0; r < (*op)->num_regions; ++r) {
        PetscCall(VecDestroy(&(*op)->petsc.material_properties[p][r]));
      }
      PetscFree((*op)->petsc.material_properties[p]);
    }
    PetscFree((*op)->petsc.material_properties);
    PetscCall(PetscOperatorDestroy(&(*op)->petsc.composite));
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
    // FIXME: check that this is set up properly
    CeedContextFieldLabel label;
    PetscCallCEED(CeedOperatorGetContextFieldLabel(op->ceed.composite, "time step", &label));
    PetscCallCEED(CeedOperatorSetContextDouble(op->ceed.composite, label, &op->ceed.dt));
  }

  {
    // The computation of fluxes across internal and boundary edges via CeedOperator is done
    // in the following three stages:
    //
    // a) Pre-CeedOperatorApply stage:
    //    - Set memory pointer of a CeedVector (u_local_ceed) is set to PETSc Vec (u_local)
    //    - Ask DM to "get" a PETSc Vec (f_local), then set memory pointer of a CeedVector (f_ceed)
    //      to the f_local PETSc Vec.
    //
    // b) CeedOperatorApply stage
    //    - Apply the CeedOparator in which u_local_ceed is an input, while f_ceed is an output.
    //
    // c) Post-CeedOperatorApply stage:
    //    - Add values in f_local to f_global via Local-to-Global scatter.
    //    - Clean up memory

    PetscScalar *u_local_ptr, *f_local_ptr;
    PetscMemType mem_type;
    Vec          f_local;

    CeedVector u_local_ceed = op->ceed.u_local;
    CeedVector f_ceed       = op->ceed.rhs;

    // 1. Sets the pointer of a CeedVector to a PETSc Vec: u_local_ceed --> u_local
    PetscCall(VecGetArrayAndMemType(u_local, &u_local_ptr, &mem_type));
    PetscCallCEED(CeedVectorSetArray(u_local_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, u_local_ptr));

    // 2. Sets the pointer of a CeedVector to a PETSc Vec: f_ceed --> f_local
    PetscCall(DMGetLocalVector(op->dm, &f_local));
    PetscCall(VecGetArrayAndMemType(f_local, &f_local_ptr, &mem_type));
    PetscCallCEED(CeedVectorSetArray(f_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, f_local_ptr));

    // 3. Apply the CeedOpeator associated with the internal and boundary edges
    PetscCall(PetscLogEventBegin(RDY_CeedOperatorApply_, u_local, f_global, 0, 0));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCEED(CeedOperatorApply(op->ceed.composite, u_local_ceed, f_ceed, CEED_REQUEST_IMMEDIATE));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogEventEnd(RDY_CeedOperatorApply_, u_local, f_global, 0, 0));

    // 4. Resets memory pointer of CeedVectors
    PetscCallCEED(CeedVectorTakeArray(f_ceed, MemTypeP2C(mem_type), &f_local_ptr));
    PetscCallCEED(CeedVectorTakeArray(u_local_ceed, MemTypeP2C(mem_type), &u_local_ptr));

    // 5. Restore pointers to the PETSc Vecs
    PetscCall(VecRestoreArrayAndMemType(f_local, &f_local_ptr));
    PetscCall(VecRestoreArrayAndMemType(u_local, &u_local_ptr));

    // 6. Zero out values in F and then add f_local to F via Local-to-Global scatter
    PetscCall(VecZeroEntries(f_global));
    PetscCall(DMLocalToGlobal(op->dm, f_local, ADD_VALUES, f_global));

    // 7. Restore the F_local
    PetscCall(DMRestoreLocalVector(op->dm, &f_local));
  }

  // FIXME: the stuff below is now bogus--both prior operators are rolled into one composite
  /*
  {
    // The computation of contribution of the source-sink term via CeedOperator is done
    // in the following three stages:
    //
    // a) Pre-CeedOperatorApply stage:
    //    - Set memory pointer of a CeedVector (u_local_ceed) to PETSc Vec (U_local)
    //    - A copy of the PETSc Vec F is made as host_fluxes. Then, memory pointer of a CeedVector (riemannf_ceed)
    //      to host_fluxes.
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
    SWESourceOperatorGetRiemannFlux(rdy->ceed.source_operator, &riemannf_field);
    PetscCallCEED(CeedOperatorFieldGetVector(riemannf_field, &riemannf_ceed));

    PetscScalar *u_local_ptr, *f_global_ptr, *host_fluxes;
    PetscMemType mem_type;
    CeedVector   u_local_ceed = rdy->ceed.u_local;
    CeedVector   s_ceed       = rdy->ceed.sources;

    // 2. Sets the pointer of a CeedVector to a PETSc Vec: u_local_ceed --> U_local
    PetscCall(VecGetArrayAndMemType(u_local, &u_local_ptr, &mem_type));
    PetscCallCEED(CeedVectorSetArray(u_local_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, u));

    // 3. Make a duplicate copy of the F as the values will be used as input for the CeedOperator
    //    corresponding to the source-sink term
    PetscCall(VecCopy(f_global, op->ceed.host_fluxes));

    // 4. Sets the pointer of a CeedVector to a PETSc Vec: host_fluxes --> riemannf_ceed
    PetscCall(VecGetArrayAndMemType(op->ceed.host_fluxes, &host_fluxes, &mem_type));
    PetscCallCEED(CeedVectorSetArray(riemannf_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, host_fluxes));

    // 5. Sets the pointer of a CeedVector to a PETSc Vec: F --> s_ceed
    PetscCall(VecGetArrayAndMemType(f_global, &f_global_ptr, &mem_type));
    PetscCallCEED(CeedVectorSetArray(s_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, f));

    // 6. Apply the source CeedOperator
    PetscCall(PetscLogEventBegin(RDY_CeedOperatorApply_, u_local, f_global, 0, 0));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCEED(CeedOperatorApply(rdy->ceed.source_operator, u_local_ceed, s_ceed, CEED_REQUEST_IMMEDIATE));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogEventEnd(RDY_CeedOperatorApply_, u_local, f_global, 0, 0));

    // 7. Reset memory pointer of CeedVectors
    PetscCallCEED(CeedVectorTakeArray(s_ceed, MemTypeP2C(mem_type), &f));
    PetscCallCEED(CeedVectorTakeArray(riemannf_ceed, MemTypeP2C(mem_type), &host_fluxes));
    PetscCallCEED(CeedVectorTakeArray(u_local_ceed, MemTypeP2C(mem_type), &u));

    // 8. Restore pointers to the PETSc Vecs
    PetscCall(VecRestoreArrayAndMemType(U_local, &u));
    PetscCall(VecRestoreArrayAndMemType(F, &f));
  }
  */

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ApplyPetscOperator(Operator *op, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBegin;
  PetscCall(PetscOperatorApply(op->petsc.composite, dt, u_local, f_global));
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

// FIXME: the CEED Courant number logic below makes assumptions about the
// FIXME: structure of our composite operator that are valid for the shallow
// FIXME: water equations, but we might need to change it to suit more general
// FIXME: conditions

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
    PetscCall(CeedFindMaxCourantNumber(op->ceed.composite, op->mesh, op->num_boundaries, op->boundaries, comm, courant_num_diags));
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
    PetscCall(GetCeedOperatorRegionData(op, region, "swe_src", source_data));
  } else {  // petsc
    PetscCall(GetPetscOperatorRegionData(op, region, op->petsc.external_sources[region.index], source_data));
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
    PetscCallCEED(RestoreCeedOperatorRegionData(op, region, "swe_src", source_data));
  } else {
    PetscCallCEED(RestorePetscOperatorRegionData(op, region, op->petsc.external_sources[region.index], source_data));
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

  // FIXME: only single-component material property data currently supported!
  property_data->num_components = 1;
  PetscCall(PetscCalloc1(property_data->num_components, &property_data->values));

  switch (property_id) {
    case OPERATOR_MANNINGS:
      if (CeedEnabled()) {
        PetscCall(GetCeedOperatorRegionData(op, region, material_property_names[property_id], property_data));
      } else {
        PetscCall(GetPetscOperatorRegionData(op, region, op->petsc.material_properties[region.index][property_id], property_data));
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
        PetscCall(RestoreCeedOperatorRegionData(op, region, material_property_names[property_id], property_data));
      } else {
        PetscCall(RestorePetscOperatorRegionData(op, region, op->petsc.material_properties[region.index][property_id], property_data));
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
PetscErrorCode PetscOperatorCreate(void          *context, PetscErrorCode (*apply)(void *, PetscReal, Vec, Vec), PetscErrorCode (*destroy)(void *),
                                   PetscOperator *op) {
  PetscFunctionBegin;
  PetscCall(PetscCalloc1(1, &op));  // NOLINT(bugprone-sizeof-expression)
  (*op)->context = context;
  (*op)->apply   = apply;
  (*op)->destroy = destroy;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Releases resources allocated to the given PetscOperator.
/// @param [inout] op the PetscOperator to be destroyed
PetscErrorCode PetscOperatorDestroy(PetscOperator *op) {
  PetscFunctionBegin;
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
  PetscCall(op->apply(op->context, dt, u_local, f_global));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  PetscInt       num_suboperators, capacity;
  PetscOperator *suboperators;
} PetscCompositeOperator;

static PetscErrorCode PetscCompositeOperatorApply(void *context, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBegin;
  PetscCompositeOperator *composite = context;
  for (PetscInt i = 0; i < composite->num_suboperators; ++i) {
    PetscOperatorApply(composite->suboperators[i], dt, u_local, f_global);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscCompositeOperatorDestroy(void *context) {
  PetscFunctionBegin;
  PetscCompositeOperator *composite = context;
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
  static const PetscInt initial_capacity = 16;
  composite->capacity                    = initial_capacity;
  PetscCall(PetscCalloc1(initial_capacity, &composite->suboperators));  // NOLINT(bugprone-sizeof-expression)
  PetscCall(PetscOperatorCreate(composite, PetscCompositeOperatorApply, PetscCompositeOperatorDestroy, op));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Appends a sub-operator to the given composite PetscOperator.
/// @param [inout] op     the composite operator to which a sub-operator is appended
/// @param [in]    sub_op the suboperator to be appended to the composite operator
PetscErrorCode PetscCompositeOperatorAddSub(PetscOperator op, PetscOperator sub_op) {
  PetscFunctionBegin;
  PetscCompositeOperator *composite = op->context;
  if (composite->num_suboperators + 1 > composite->capacity) {
    composite->capacity *= 2;
    PetscCall(PetscRealloc(composite->capacity, &composite->suboperators));
  }
  composite->suboperators[composite->num_suboperators] = sub_op;
  ++composite->num_suboperators;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
