#include <petscdmceed.h>
#include <private/rdycoreimpl.h>
#include <private/rdymathimpl.h>
#include <private/rdysweimpl.h>
#include <stddef.h>  // for offsetof

PetscClassId  RDY_CLASSID;
PetscLogEvent RDY_CeedOperatorApply;

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

  // create an MPI data type for the CourantNumberDiagnostics struct
  const int      num_blocks             = 3;
  const int      block_lengths[3]       = {1, 1, 1};
  const MPI_Aint block_displacements[3] = {
      offsetof(CourantNumberDiagnostics, max_courant_num),
      offsetof(CourantNumberDiagnostics, global_edge_id),
      offsetof(CourantNumberDiagnostics, global_cell_id),
  };
  MPI_Datatype block_types[3] = {MPI_DOUBLE, MPI_INT, MPI_INT};
  MPI_Type_create_struct(num_blocks, block_lengths, block_displacements, block_types, &courant_num_diags_type);
  MPI_Type_commit(&courant_num_diags_type);

  // create a corresponding reduction operator for the new type
  MPI_Op_create(FindCourantNumberDiagnostics, 1, &courant_num_diags_op);

  // make sure the operator and the type are destroyed upon exit
  PetscCall(RDyOnFinalize(DestroyCourantNumberDiagnostics));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// create solvers and vectors
static PetscErrorCode CreateSolvers(RDy rdy) {
  PetscFunctionBegin;

  if (!rdy->ceed_resource[0]) {
    // swe_src is only needed for PETSc source operator
    PetscCall(VecDuplicate(rdy->X, &rdy->swe_src));
    PetscCall(VecZeroEntries(rdy->swe_src));
  }

  PetscInt n_dof;
  PetscCall(VecGetSize(rdy->X, &n_dof));

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

  PetscCheck(rdy->config.physics.flow.mode == FLOW_SWE, rdy->comm, PETSC_ERR_USER, "Only the 'swe' flow mode is currently supported.");
  PetscCall(TSSetRHSFunction(rdy->ts, rdy->R, RHSFunctionSWE, rdy));

  PetscCall(TSSetMaxSteps(rdy->ts, rdy->config.time.max_step));
  PetscCall(TSSetExactFinalTime(rdy->ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetSolution(rdy->ts, rdy->X));
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
  if (rdy->ceed_resource[0]) {
    RDyLogDebug(rdy, "Setting up CEED Operators...");

    // create the operators themselves
    PetscCall(CreateSWEFluxOperator(rdy->ceed, &rdy->mesh, rdy->num_boundaries, rdy->boundaries, rdy->boundary_conditions,
                                    rdy->config.physics.flow.tiny_h, &rdy->ceed_rhs.op_edges));

    PetscCall(CreateSWESourceOperator(rdy->ceed, &rdy->mesh, rdy->mesh.num_cells, rdy->materials_by_cell, rdy->config.physics.flow.tiny_h,
                                      &rdy->ceed_rhs.op_src));

    // create associated vectors for storage
    int num_comp = 3;
    PetscCallCEED(CeedVectorCreate(rdy->ceed, rdy->mesh.num_cells * num_comp, &rdy->ceed_rhs.u_local_ceed));
    PetscCallCEED(CeedVectorCreate(rdy->ceed, rdy->mesh.num_cells * num_comp, &rdy->ceed_rhs.f_ceed));
    PetscCallCEED(CeedVectorCreate(rdy->ceed, rdy->mesh.num_cells_local * num_comp, &rdy->ceed_rhs.s_ceed));
    PetscCallCEED(CeedVectorCreate(rdy->ceed, rdy->mesh.num_cells_local * num_comp, &rdy->ceed_rhs.u_ceed));

    // reset the time step size
    rdy->ceed_rhs.dt = 0.0;
  } else {
    // allocate storage for our PETSc implementation of the  flux and
    // source terms
    RDyLogDebug(rdy, "Allocating PETSc data structures for fluxes and sources...");
    PetscCall(CreatePetscSWEFlux(rdy->mesh.num_internal_edges, rdy->num_boundaries, rdy->boundaries, &rdy->petsc_rhs));
    PetscCall(CreatePetscSWESource(&rdy->mesh, rdy->petsc_rhs));
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
  PetscCall(PetscClassIdRegister("RDycore", &RDY_CLASSID));
  PetscCall(PetscLogEventRegister("CeedOperatorApp", RDY_CLASSID, &RDY_CeedOperatorApply));

  // sets up solvers, operators, all that stuff
  PetscCall(CreateSolvers(rdy));
  PetscCall(CreateOperators(rdy));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline CeedMemType MemTypeP2C(PetscMemType mem_type) { return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST; }

static PetscErrorCode RDyCeedOperatorApply(RDy rdy, PetscReal dt, Vec U_local, Vec F) {
  PetscFunctionBeginUser;

  // update the timestep for the ceed operators if necessary
  if (rdy->ceed_rhs.dt != dt) {
    PetscCall(SWEFluxOperatorSetTimeStep(rdy->ceed_rhs.op_edges, dt));
    PetscCall(SWESourceOperatorSetTimeStep(rdy->ceed_rhs.op_src, dt));
    rdy->ceed_rhs.dt = dt;
  }

  {
    PetscScalar *u_local, *f;
    PetscMemType mem_type;
    Vec          F_local;

    CeedVector u_local_ceed = rdy->ceed_rhs.u_local_ceed;
    CeedVector f_ceed       = rdy->ceed_rhs.f_ceed;

    PetscCall(VecGetArrayAndMemType(U_local, &u_local, &mem_type));
    PetscCallCEED(CeedVectorSetArray(u_local_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, u_local));

    PetscCall(DMGetLocalVector(rdy->dm, &F_local));
    PetscCall(VecGetArrayAndMemType(F_local, &f, &mem_type));
    PetscCallCEED(CeedVectorSetArray(f_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, f));

    PetscCall(PetscLogEventBegin(RDY_CeedOperatorApply, U_local, F, 0, 0));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCEED(CeedOperatorApply(rdy->ceed_rhs.op_edges, u_local_ceed, f_ceed, CEED_REQUEST_IMMEDIATE));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogEventEnd(RDY_CeedOperatorApply, U_local, F, 0, 0));

    PetscCallCEED(CeedVectorTakeArray(u_local_ceed, MemTypeP2C(mem_type), &u_local));
    PetscCall(VecRestoreArrayAndMemType(U_local, &u_local));

    PetscCallCEED(CeedVectorTakeArray(f_ceed, MemTypeP2C(mem_type), &f));
    PetscCall(VecRestoreArrayAndMemType(F_local, &f));

    PetscCall(VecZeroEntries(F));
    PetscCall(DMLocalToGlobal(rdy->dm, F_local, ADD_VALUES, F));
    PetscCall(DMRestoreLocalVector(rdy->dm, &F_local));
  }

  {
    CeedOperatorField riemannf_field;
    SWESourceOperatorGetRiemannFlux(rdy->ceed_rhs.op_src, &riemannf_field);

    CeedVector riemannf_ceed;
    PetscCallCEED(CeedOperatorFieldGetVector(riemannf_field, &riemannf_ceed));

    PetscScalar *u, *f, *f_dup;
    PetscMemType mem_type;
    CeedVector   u_ceed = rdy->ceed_rhs.u_ceed;
    CeedVector   s_ceed = rdy->ceed_rhs.s_ceed;

    PetscCall(VecGetArrayAndMemType(rdy->Soln, &u, &mem_type));
    PetscCallCEED(CeedVectorSetArray(u_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, u));

    Vec F_dup;
    PetscCall(VecDuplicate(F, &F_dup));
    PetscCall(VecCopy(F, F_dup));
    PetscCall(VecGetArrayAndMemType(F_dup, &f_dup, &mem_type));
    PetscCallCEED(CeedVectorSetArray(riemannf_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, f_dup));

    PetscCall(VecGetArrayAndMemType(F, &f, &mem_type));
    PetscCallCEED(CeedVectorSetArray(s_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, f));

    PetscCall(PetscLogEventBegin(RDY_CeedOperatorApply, U_local, F, 0, 0));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCEED(CeedOperatorApply(rdy->ceed_rhs.op_src, u_ceed, s_ceed, CEED_REQUEST_IMMEDIATE));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogEventEnd(RDY_CeedOperatorApply, U_local, F, 0, 0));

    PetscCallCEED(CeedVectorTakeArray(u_ceed, MemTypeP2C(mem_type), &u));
    PetscCall(VecRestoreArrayAndMemType(rdy->Soln, &u));
    PetscCallCEED(CeedVectorTakeArray(riemannf_ceed, MemTypeP2C(mem_type), &f_dup));
    PetscCallCEED(CeedVectorTakeArray(s_ceed, MemTypeP2C(mem_type), &f));
    PetscCall(VecRestoreArrayAndMemType(F, &f));
    PetscCall(VecDestroy(&F_dup));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// This is the right-hand-side function used by our timestepping solver for
// the shallow water equations.
// Parameters:
//  ts  - the solver
//  t   - the simulation time [seconds]
//  X   - the solution vector at time t
//  F   - the right hand side vector to be evaluated at time t
//  ctx - a generic pointer to our RDy object
PetscErrorCode RHSFunctionSWE(TS ts, PetscReal t, Vec X, Vec F, void *ctx) {
  PetscFunctionBegin;

  RDy rdy = ctx;
  DM  dm  = rdy->dm;

  PetscScalar dt;
  PetscCall(TSGetTimeStep(ts, &dt));

  PetscCall(VecZeroEntries(F));

  // populate the local X vector
  PetscCall(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, rdy->X_local));
  PetscCall(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, rdy->X_local));

  // compute the right hand side
  CourantNumberDiagnostics courant_num_diags = {
      .max_courant_num = 0.0,
      .global_edge_id  = -1,
      .global_cell_id  = -1,
  };

  if (rdy->ceed_resource[0]) {
    PetscCall(VecCopy(X, rdy->Soln));
    PetscCall(RDyCeedOperatorApply(rdy, dt, rdy->X_local, F));
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
    PetscCall(SWERHSFunctionForInternalEdges(rdy, F, &courant_num_diags));
    PetscCall(SWERHSFunctionForBoundaryEdges(rdy, F, &courant_num_diags));
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

  // write out debugging info for maximum courant number
  if (rdy->config.logging.level >= LOG_DEBUG) {
    MPI_Allreduce(MPI_IN_PLACE, &courant_num_diags, 1, courant_num_diags_type, courant_num_diags_op, rdy->comm);
    PetscReal time;
    PetscInt  stepnum;
    PetscCall(TSGetTime(ts, &time));
    PetscCall(TSGetStepNumber(ts, &stepnum));
    RDyLogDebug(rdy, "[%" PetscInt_FMT "] Time = %f Max courant number %g encountered at edge %" PetscInt_FMT " of cell %" PetscInt_FMT " is %f",
                stepnum, time, courant_num_diags.max_courant_num, courant_num_diags.global_edge_id, courant_num_diags.global_cell_id,
                courant_num_diags.max_courant_num);
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
