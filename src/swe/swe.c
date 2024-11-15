#include <petscdmceed.h>
#include <private/rdycoreimpl.h>
#include <private/rdymathimpl.h>
#include <private/rdysweimpl.h>
#include <stddef.h>  // for offsetof

extern PetscLogEvent RDY_CeedOperatorApply;

// these functions are implemented in swe_flux_petsc.c
PetscErrorCode SWERHSFunctionForInternalEdges(RDy rdy, Vec F, CourantNumberDiagnostics *courant_num_diags);
PetscErrorCode SWERHSFunctionForBoundaryEdges(RDy rdy, Vec F, CourantNumberDiagnostics *courant_num_diags);
PetscErrorCode ComputeSWEDiagnosticVariables(RDy rdy);
PetscErrorCode AddSWESourceTerm(RDy rdy, Vec F);

//-----------------------
// Debugging diagnostics
//-----------------------

// MPI datatype corresponding to CourantNumberDiagnostics. Created during InitSWE.
static MPI_Datatype courant_num_diags_type;

// MPI operator used to determine the prevailing diagnostics for the maximum
// courant number on all processes. Created during InitSWE.
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

// this function is called by InitSWE to initialize the above type(s) and op(s).
static PetscErrorCode InitMPITypesAndOps(void) {
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

static PetscErrorCode RHSFunctionSWE(TS, PetscReal, Vec, Vec, void *);

// create solvers and vectors
static PetscErrorCode CreateSolvers(RDy rdy) {
  PetscFunctionBegin;

  PetscInt n_dof;
  PetscCall(VecGetSize(rdy->u_global, &n_dof));

  // set up a TS solver
  PetscCall(TSCreate(rdy->comm, &rdy->ts));
  PetscCall(TSSetProblemType(rdy->ts, TS_NONLINEAR));
  switch (rdy->config.numerics.temporal) {
    case TEMPORAL_EULER:
      PetscCall(TSSetType(rdy->ts, TSEULER));
      break;
    case TEMPORAL_RK4:
      PetscCall(TSSetType(rdy->ts, TSRK));
      PetscCall(TSRKSetType(rdy->ts, TSRK4));
      break;
    case TEMPORAL_BEULER:
      PetscCall(TSSetType(rdy->ts, TSBEULER));
      break;
  }
  PetscCall(TSSetDM(rdy->ts, rdy->dm));
  PetscCall(TSSetApplicationContext(rdy->ts, rdy));

  PetscCheck(rdy->config.physics.flow.mode == FLOW_SWE, rdy->comm, PETSC_ERR_USER, "Only the 'swe' flow mode is currently supported.");
  PetscCall(TSSetRHSFunction(rdy->ts, rdy->rhs, RHSFunctionSWE, rdy));

  if (!rdy->config.time.adaptive.enable) {
    PetscCall(TSSetMaxSteps(rdy->ts, rdy->config.time.max_step));
  }
  PetscCall(TSSetExactFinalTime(rdy->ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetSolution(rdy->ts, rdy->u_global));
  PetscCall(TSSetTime(rdy->ts, 0.0));
  PetscCall(TSSetTimeStep(rdy->ts, rdy->dt));

  // apply any solver-related options supplied on the command line
  PetscCall(TSSetFromOptions(rdy->ts));
  PetscCall(TSGetTimeStep(rdy->ts, &rdy->dt));  // just in case!

  PetscFunctionReturn(PETSC_SUCCESS);
}

// create flux and source operators
static PetscErrorCode CreateOperators(RDy rdy) {
  PetscFunctionBegin;
  if (CeedEnabled()) {
    RDyLogDebug(rdy, "Setting up CEED Operators...");

    // create the operators themselves
    Ceed ceed = CeedContext();
    PetscCall(CreateSWEFluxOperator(ceed, &rdy->mesh, rdy->num_boundaries, rdy->boundaries, rdy->boundary_conditions, rdy->config.physics.flow.tiny_h,
                                    &rdy->ceed.flux_operator));

    PetscCall(CreateSWESourceOperator(ceed, &rdy->mesh, rdy->mesh.num_cells, rdy->materials_by_cell, rdy->config.physics.flow.tiny_h,
                                      &rdy->ceed.source_operator));

    // create associated vectors for storage
    int num_comp = 3;
    PetscCallCEED(CeedVectorCreate(ceed, rdy->mesh.num_cells * num_comp, &rdy->ceed.u_local));
    PetscCallCEED(CeedVectorCreate(ceed, rdy->mesh.num_cells * num_comp, &rdy->ceed.rhs));
    PetscCallCEED(CeedVectorCreate(ceed, rdy->mesh.num_owned_cells * num_comp, &rdy->ceed.sources));

    PetscBool ceed_enabled = PETSC_TRUE;
    PetscCall(
        CreatePetscSWEFluxForBoundaryEdges(&rdy->mesh.edges, num_comp, rdy->num_boundaries, rdy->boundaries, ceed_enabled, &rdy->petsc.context));

    // reset the time step size
    rdy->ceed.dt = 0.0;
  } else {
    // allocate storage for our PETSc implementation of the  flux and
    // source terms
    RDyLogDebug(rdy, "Allocating PETSc data structures for fluxes and sources...");
    int       num_comp     = 3;
    PetscBool ceed_enabled = PETSC_FALSE;
    PetscCall(CreatePetscSWEFluxForInternalEdges(&rdy->mesh.edges, num_comp, rdy->mesh.num_internal_edges, &rdy->petsc.context));
    PetscCall(
        CreatePetscSWEFluxForBoundaryEdges(&rdy->mesh.edges, num_comp, rdy->num_boundaries, rdy->boundaries, ceed_enabled, &rdy->petsc.context));
    PetscCall(CreatePetscSWESource(&rdy->mesh, rdy->petsc.context));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

//---------------------------
// End debugging diagnostics
//---------------------------

// This function initializes SWE physics for the given dycore.
PetscErrorCode InitSWE(RDy rdy) {
  PetscFunctionBeginUser;

  // set up MPI types and operators used by SWE physics
  PetscCall(InitMPITypesAndOps());

  // sets up solvers, operators, all that stuff
  PetscCall(CreateSolvers(rdy));
  PetscCall(CreateOperators(rdy));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline CeedMemType MemTypeP2C(PetscMemType mem_type) { return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST; }

static PetscErrorCode RDyCeedOperatorApply(RDy rdy, PetscReal dt, Vec U_local, Vec F) {
  PetscFunctionBeginUser;

  // update the timestep for the ceed operators if necessary
  if (rdy->ceed.dt != dt) {
    PetscCall(SWEFluxOperatorSetTimeStep(rdy->ceed.flux_operator, dt));
    PetscCall(SWESourceOperatorSetTimeStep(rdy->ceed.source_operator, dt));
    rdy->ceed.dt = dt;
  }

  {
    // The computation of fluxes across internal and boundary edges via CeedOperator is done
    // in the following three stages:
    //
    // a) Pre-CeedOperatorApply stage:
    //    - Set memory pointer of a CeedVector (u_local_ceed) is set to PETSc Vec (U_local)
    //    - Ask DM to "get" a PETSc Vec (F_local), then set memory pointer of a CeedVector (f_ceed)
    //      to the F_local PETSc Vec.
    //
    // b) CeedOperatorApply stage
    //    - Apply the CeedOparator in which u_local_ceed is an input, while f_ceed is an output.
    //
    // c) Post-CeedOperatorApply stage:
    //    - Add values in F_local to F via Local-to-Global scatter.
    //    - Clean up memory

    PetscScalar *u_local, *f;
    PetscMemType mem_type;
    Vec          F_local;

    CeedVector u_local_ceed = rdy->ceed.u_local;
    CeedVector f_ceed       = rdy->ceed.rhs;

    // 1. Sets the pointer of a CeedVector to a PETSc Vec: u_local_ceed --> U_local
    PetscCall(VecGetArrayAndMemType(U_local, &u_local, &mem_type));
    PetscCallCEED(CeedVectorSetArray(u_local_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, u_local));

    // 2. Sets the pointer of a CeedVector to a PETSc Vec: f_ceed --> F_local
    PetscCall(DMGetLocalVector(rdy->dm, &F_local));
    PetscCall(VecGetArrayAndMemType(F_local, &f, &mem_type));
    PetscCallCEED(CeedVectorSetArray(f_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, f));

    // 3. Apply the CeedOpeator associated with the internal and boundary edges
    PetscCall(PetscLogEventBegin(RDY_CeedOperatorApply, U_local, F, 0, 0));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCEED(CeedOperatorApply(rdy->ceed.flux_operator, u_local_ceed, f_ceed, CEED_REQUEST_IMMEDIATE));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogEventEnd(RDY_CeedOperatorApply, U_local, F, 0, 0));

    // 4. Resets memory pointer of CeedVectors
    PetscCallCEED(CeedVectorTakeArray(f_ceed, MemTypeP2C(mem_type), &f));
    PetscCallCEED(CeedVectorTakeArray(u_local_ceed, MemTypeP2C(mem_type), &u_local));

    // 5. Restore pointers to the PETSc Vecs
    PetscCall(VecRestoreArrayAndMemType(F_local, &f));
    PetscCall(VecRestoreArrayAndMemType(U_local, &u_local));

    // 6. Zero out values in F and then add F_local to F via Local-to-Global scatter
    PetscCall(VecZeroEntries(F));
    PetscCall(DMLocalToGlobal(rdy->dm, F_local, ADD_VALUES, F));

    // 7. Restor the F_local
    PetscCall(DMRestoreLocalVector(rdy->dm, &F_local));
  }

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

    PetscScalar *u, *f, *host_fluxes;
    PetscMemType mem_type;
    CeedVector   u_local_ceed = rdy->ceed.u_local;
    CeedVector   s_ceed       = rdy->ceed.sources;

    // 2. Sets the pointer of a CeedVector to a PETSc Vec: u_local_ceed --> U_local
    PetscCall(VecGetArrayAndMemType(U_local, &u, &mem_type));
    PetscCallCEED(CeedVectorSetArray(u_local_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, u));

    // 3. Make a duplicate copy of the F as the values will be used as input for the CeedOperator
    //    corresponding to the source-sink term
    PetscCall(VecCopy(F, rdy->ceed.host_fluxes));

    // 4. Sets the pointer of a CeedVector to a PETSc Vec: host_fluxes --> riemannf_ceed
    PetscCall(VecGetArrayAndMemType(rdy->ceed.host_fluxes, &host_fluxes, &mem_type));
    PetscCallCEED(CeedVectorSetArray(riemannf_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, host_fluxes));

    // 5. Sets the pointer of a CeedVector to a PETSc Vec: F --> s_ceed
    PetscCall(VecGetArrayAndMemType(F, &f, &mem_type));
    PetscCallCEED(CeedVectorSetArray(s_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, f));

    // 6. Apply the source CeedOperator
    PetscCall(PetscLogEventBegin(RDY_CeedOperatorApply, U_local, F, 0, 0));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCEED(CeedOperatorApply(rdy->ceed.source_operator, u_local_ceed, s_ceed, CEED_REQUEST_IMMEDIATE));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogEventEnd(RDY_CeedOperatorApply, U_local, F, 0, 0));

    // 7. Reset memory pointer of CeedVectors
    PetscCallCEED(CeedVectorTakeArray(s_ceed, MemTypeP2C(mem_type), &f));
    PetscCallCEED(CeedVectorTakeArray(riemannf_ceed, MemTypeP2C(mem_type), &host_fluxes));
    PetscCallCEED(CeedVectorTakeArray(u_local_ceed, MemTypeP2C(mem_type), &u));

    // 8. Restore pointers to the PETSc Vecs
    PetscCall(VecRestoreArrayAndMemType(U_local, &u));
    PetscCall(VecRestoreArrayAndMemType(F, &f));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Loops over all internal edges and finds the local maximum courant number.
///        If needed, the data is moved from device to host.
/// @param [in] op_edges A CeedOperator object for edges
/// @param [in] mesh A pointer to a RDyMesh object
/// @param [in] *max_courant_number Local maximum value of courant number
/// @return 0 on sucess, or a non-zero error code on failure
static PetscErrorCode CeedFindMaxCourantNumberInternalEdges(CeedOperator op_edges, RDyMesh *mesh, PetscReal *max_courant_number) {
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
    CeedScalar local_max = fmax(courant_num_data[ii][0], courant_num_data[ii][1]);
    *max_courant_number  = fmax(*max_courant_number, local_max);
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
                                                            PetscReal *max_courant_number) {
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
      *max_courant_number = fmax(*max_courant_number, courant_num_data[e][0]);
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
                                               PetscReal *max_courant_number) {
  PetscFunctionBegin;

  PetscCall(CeedFindMaxCourantNumberInternalEdges(op_edges, mesh, max_courant_number));
  PetscCall(CeedFindMaxCourantNumberBoundaryEdges(op_edges, num_boundaries, boundaries, max_courant_number));

  PetscCall(MPI_Allreduce(MPI_IN_PLACE, max_courant_number, 1, MPI_DOUBLE, MPI_MAX, comm));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Finds the maximum Courant number for the libCEED and the PETSc version of SWE implementation
/// @param [inout] rdy An RDy object
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode SWEFindMaxCourantNumber(RDy rdy) {
  PetscFunctionBegin;

  CourantNumberDiagnostics *courant_num_diags = &rdy->courant_num_diags;

  if (CeedEnabled()) {
    PetscCall(CeedFindMaxCourantNumber(rdy->ceed.flux_operator, &rdy->mesh, rdy->num_boundaries, rdy->boundaries, rdy->comm,
                                       &courant_num_diags->max_courant_num));
    courant_num_diags->is_set = PETSC_TRUE;
  } else {
    MPI_Allreduce(MPI_IN_PLACE, courant_num_diags, 1, courant_num_diags_type, courant_num_diags_op, rdy->comm);
    courant_num_diags->is_set = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// This is the right-hand-side function used by our timestepping solver for
// the shallow water equations.
// Parameters:
//  ts  - the solver
//  t   - the simulation time [seconds]
//  U   - the solution vector at time t
//  F   - the right hand side vector to be evaluated at time t
//  ctx - a generic pointer to our RDy object
PetscErrorCode RHSFunctionSWE(TS ts, PetscReal t, Vec U, Vec F, void *ctx) {
  PetscFunctionBegin;

  RDy rdy = ctx;
  DM  dm  = rdy->dm;

  PetscScalar dt;
  PetscCall(TSGetTimeStep(ts, &dt));

  PetscCall(VecZeroEntries(F));

  // populate the local U vector
  PetscCall(DMGlobalToLocalBegin(dm, U, INSERT_VALUES, rdy->u_local));
  PetscCall(DMGlobalToLocalEnd(dm, U, INSERT_VALUES, rdy->u_local));

  // get courant number diagnostics
  CourantNumberDiagnostics *courant_num_diags = &rdy->courant_num_diags;
  courant_num_diags->max_courant_num          = 0.0;

  // compute the right hand side
  if (CeedEnabled()) {
    PetscCall(RDyCeedOperatorApply(rdy, dt, rdy->u_local, F));
    if (0) {
      PetscInt nstep;
      PetscCall(TSGetStepNumber(ts, &nstep));

      char file[PETSC_MAX_PATH_LEN];
      sprintf(file, "F_ceed_nstep%" PetscInt_FMT "_N%d.bin", nstep, rdy->nproc);

      PetscViewer viewer;
      PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_WRITE, &viewer));
      PetscCall(VecView(F, viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    }
  } else {
    PetscCall(ComputeSWEDiagnosticVariables(rdy));
    PetscCall(SWERHSFunctionForInternalEdges(rdy, F, courant_num_diags));
    PetscCall(SWERHSFunctionForBoundaryEdges(rdy, F, courant_num_diags));
    PetscCall(AddSWESourceTerm(rdy, F));
    if (0) {
      PetscInt nstep;
      PetscCall(TSGetStepNumber(ts, &nstep));

      char file[PETSC_MAX_PATH_LEN];
      sprintf(file, "F_petsc_nstep%" PetscInt_FMT "_N%d.bin", nstep, rdy->nproc);

      PetscViewer viewer;
      PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_WRITE, &viewer));
      PetscCall(VecView(F, viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    }
  }

  // if debug-level logging is enabled, find the latest global maximum Courant number
  // and log it.
  if (rdy->config.logging.level >= LOG_DEBUG) {
    PetscCall(SWEFindMaxCourantNumber(rdy));

    PetscReal time;
    PetscInt  stepnum;
    PetscCall(TSGetTime(ts, &time));
    PetscCall(TSGetStepNumber(ts, &stepnum));
    const char *units = TimeUnitAsString(rdy->config.time.unit);

    RDyLogDebug(rdy, "[%" PetscInt_FMT "] Time = %f [%s] Max courant number %g", stepnum, ConvertTimeFromSeconds(time, rdy->config.time.unit), units,
                courant_num_diags->max_courant_num);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RiemannDataSWECreate(PetscInt N, RiemannDataSWE *data) {
  PetscFunctionBegin;

  data->N = N;
  PetscCall(PetscCalloc1(data->N, &data->h));
  PetscCall(PetscCalloc1(data->N, &data->hu));
  PetscCall(PetscCalloc1(data->N, &data->hv));
  PetscCall(PetscCalloc1(data->N, &data->u));
  PetscCall(PetscCalloc1(data->N, &data->v));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RiemannDataSWEDestroy(RiemannDataSWE data) {
  PetscFunctionBegin;

  data.N = 0;
  PetscCall(PetscFree(data.h));
  PetscCall(PetscFree(data.hu));
  PetscCall(PetscFree(data.hv));
  PetscCall(PetscFree(data.u));
  PetscCall(PetscFree(data.v));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RiemannEdgeDataSWECreate(PetscInt N, PetscInt ncomp, RiemannEdgeDataSWE *data) {
  PetscFunctionBegin;

  data->N = N;
  PetscCall(PetscCalloc1(data->N, &data->cn));
  PetscCall(PetscCalloc1(data->N, &data->sn));
  PetscCall(PetscCalloc1(data->N * ncomp, &data->flux));
  PetscCall(PetscCalloc1(data->N, &data->amax));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RiemannEdgeDataSWEDestroy(RiemannEdgeDataSWE data) {
  PetscFunctionBegin;

  data.N = 0;
  PetscCall(PetscFree(data.cn));
  PetscCall(PetscFree(data.sn));
  PetscCall(PetscFree(data.flux));
  PetscCall(PetscFree(data.amax));

  PetscFunctionReturn(PETSC_SUCCESS);
}
