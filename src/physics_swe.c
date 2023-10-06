#include <private/rdycoreimpl.h>
#include <private/rdymathimpl.h>
#include <private/rdymemoryimpl.h>
#include <stddef.h>  // for offsetof

#include "swe_flux_ceed.h"
#include "swe_flux_petsc.h"

PetscClassId  RDY_CLASSID;
PetscLogEvent RDY_CeedOperatorApply;

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

  PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
}

static PetscErrorCode RDyCeedOperatorUpdateDt(RDy rdy, PetscReal dt) {
  PetscFunctionBeginUser;
  if (rdy->ceed_resource[0]) {
    if (rdy->ceed_rhs.dt != dt) {
      rdy->ceed_rhs.dt = dt;

      CeedContextFieldLabel label;

      CeedOperatorGetContextFieldLabel(rdy->ceed_rhs.op_edges, "time step", &label);
      CeedOperatorSetContextDouble(rdy->ceed_rhs.op_edges, label, &dt);

      CeedOperatorGetContextFieldLabel(rdy->ceed_rhs.op_src, "time step", &label);
      CeedOperatorSetContextDouble(rdy->ceed_rhs.op_src, label, &dt);
    }
  }
  PetscFunctionReturn(0);
}

/// If the source term was updated, copy new values for libCEED
/// @param [inout] rdy A RDy struc
///
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode RDyCeedUpdateSourceTerm(RDy rdy) {
  PetscFunctionBeginUser;

  if (!rdy->ceed_rhs.water_src_updated) {
    PetscInt num_sub_ops;
    CeedCompositeOperatorGetNumSub(rdy->ceed_rhs.op_src, &num_sub_ops);

    CeedOperator *sub_ops;
    CeedCompositeOperatorGetSubList(rdy->ceed_rhs.op_src, &sub_ops);

    PetscInt          source_op_id = rdy->ceed_rhs.water_src_op_id;
    CeedOperatorField water_src_field;
    CeedOperatorGetFieldByName(sub_ops[source_op_id], "water_src", &water_src_field);
    CeedVector water_src;
    CeedOperatorFieldGetVector(water_src_field, &water_src);

    PetscInt num_comp_water_src = 1;
    CeedScalar(*wat_src_ceed)[num_comp_water_src];
    CeedVectorGetArray(water_src, CEED_MEM_HOST, (CeedScalar **)&wat_src_ceed);
    PetscScalar *wat_src_p;
    PetscCall(VecGetArray(rdy->water_src, &wat_src_p));

    for (PetscInt i = 0; i < rdy->mesh.num_cells_local; ++i) {
      wat_src_ceed[i][0] = wat_src_p[i];
    }

    CeedVectorRestoreArray(water_src, (CeedScalar **)&wat_src_ceed);
    PetscCall(VecRestoreArray(rdy->water_src, &wat_src_p));

    rdy->ceed_rhs.water_src_updated = PETSC_TRUE;
  }

  PetscFunctionReturn(0);
}

static inline CeedMemType MemTypeP2C(PetscMemType mem_type) { return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST; }

static PetscErrorCode RDyCeedOperatorApply(RDy rdy, PetscReal dt, Vec U_local, Vec F) {
  PetscFunctionBeginUser;

  PetscCall(RDyCeedOperatorUpdateDt(rdy, dt));
  PetscCall(RDyCeedUpdateSourceTerm(rdy));

  {
    PetscScalar *u_local, *f;
    PetscMemType mem_type;
    Vec          F_local;

    CeedVector u_local_ceed = rdy->ceed_rhs.u_local_ceed;
    CeedVector f_ceed       = rdy->ceed_rhs.f_ceed;

    PetscCall(VecGetArrayAndMemType(U_local, &u_local, &mem_type));
    CeedVectorSetArray(u_local_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, u_local);

    PetscCall(DMGetLocalVector(rdy->dm, &F_local));
    PetscCall(VecGetArrayAndMemType(F_local, &f, &mem_type));
    CeedVectorSetArray(f_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, f);

    PetscCall(PetscLogEventBegin(RDY_CeedOperatorApply, U_local, F, 0, 0));
    PetscCall(PetscLogGpuTimeBegin());
    CeedOperatorApply(rdy->ceed_rhs.op_edges, u_local_ceed, f_ceed, CEED_REQUEST_IMMEDIATE);
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogEventEnd(RDY_CeedOperatorApply, U_local, F, 0, 0));

    CeedVectorTakeArray(u_local_ceed, MemTypeP2C(mem_type), &u_local);
    PetscCall(VecRestoreArrayAndMemType(U_local, &u_local));

    CeedVectorTakeArray(f_ceed, MemTypeP2C(mem_type), &f);
    PetscCall(VecRestoreArrayAndMemType(F_local, &f));

    PetscCall(VecZeroEntries(F));
    PetscCall(DMLocalToGlobal(rdy->dm, F_local, ADD_VALUES, F));
    PetscCall(DMRestoreLocalVector(rdy->dm, &F_local));
  }

  {
    CeedOperator *sub_ops;
    CeedCompositeOperatorGetSubList(rdy->ceed_rhs.op_src, &sub_ops);

    PetscInt source_op_id = rdy->ceed_rhs.water_src_op_id;

    CeedOperatorField riemannf_field;
    CeedOperatorGetFieldByName(sub_ops[source_op_id], "riemannf", &riemannf_field);

    CeedVector riemannf_ceed;
    CeedOperatorFieldGetVector(riemannf_field, &riemannf_ceed);

    PetscScalar *u, *f, *f_dup;
    PetscMemType mem_type;
    CeedVector   u_ceed = rdy->ceed_rhs.u_ceed;
    CeedVector   s_ceed = rdy->ceed_rhs.s_ceed;

    PetscCall(VecGetArrayAndMemType(rdy->Soln, &u, &mem_type));
    CeedVectorSetArray(u_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, u);

    Vec F_dup;
    PetscCall(VecDuplicate(F, &F_dup));
    PetscCall(VecCopy(F, F_dup));
    PetscCall(VecGetArrayAndMemType(F_dup, &f_dup, &mem_type));
    CeedVectorSetArray(riemannf_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, f_dup);

    PetscCall(VecGetArrayAndMemType(F, &f, &mem_type));
    CeedVectorSetArray(s_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, f);

    PetscCall(PetscLogEventBegin(RDY_CeedOperatorApply, U_local, F, 0, 0));
    PetscCall(PetscLogGpuTimeBegin());
    CeedOperatorApply(rdy->ceed_rhs.op_src, u_ceed, s_ceed, CEED_REQUEST_IMMEDIATE);
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogEventEnd(RDY_CeedOperatorApply, U_local, F, 0, 0));

    CeedVectorTakeArray(u_ceed, MemTypeP2C(mem_type), &u);
    PetscCall(VecRestoreArrayAndMemType(rdy->Soln, &u));
    CeedVectorTakeArray(riemannf_ceed, MemTypeP2C(mem_type), &f_dup);
    CeedVectorTakeArray(s_ceed, MemTypeP2C(mem_type), &f);
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
      sprintf(file, "F_ceed_nstep%d_N%d.bin", nstep, rdy->nproc);

      PetscViewer viewer;
      PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_WRITE, &viewer));
      PetscCall(VecView(F, viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    }
  } else {
    PetscCall(RHSFunctionForInternalEdges(rdy, F, &courant_num_diags));
    PetscCall(RHSFunctionForBoundaryEdges(rdy, F, &courant_num_diags));
    PetscCall(AddSourceTerm(rdy, F));  // TODO: move source term to use libCEED
    if (0) {
      PetscInt nstep;
      PetscCall(TSGetStepNumber(ts, &nstep));

      char file[PETSC_MAX_PATH_LEN];
      sprintf(file, "F_petsc_nstep%d_N%d.bin", nstep, rdy->nproc);

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
    RDyLogDebug(rdy, "[%d] Time = %f Max courant number %g encountered at edge %d of cell %d is %f", stepnum, time, courant_num_diags.max_courant_num,
                courant_num_diags.global_edge_id, courant_num_diags.global_cell_id, courant_num_diags.max_courant_num);
  }

  PetscFunctionReturn(0);
}
