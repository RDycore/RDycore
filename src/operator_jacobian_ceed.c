// libCEED evaluation of the analytic Jacobian dF/dU of the SWE right-hand side for
// a fully implicit flow TS (plain Roe, hydrostatic reconstruction, or augmented Roe
// interior fluxes; see swe/swe_edge_flux_jacobian_ceed.h).
//
// A composite CeedOperator evaluates the 3x3 Jacobian blocks of every interior edge,
// boundary edge and cell (Q-functions in swe/swe_jacobian_ceed.h) into one
// contiguous COO-values CeedVector. The COO (row, col) pattern is built once on the
// host from the residual operator's element restrictions (local dof offsets), and
// handed to PETSc with MatSetPreallocationCOOLocal. Each Jacobian evaluation is then
// one CeedOperatorApply plus one MatSetValuesCOO, in the memory space of the state
// vector (host, or device for GPU backends), with no host<->device copies.
//
// The Jacobian sub-operators reuse the residual sub-operators' restrictions and
// passive vectors (geometry, Dirichlet values, material properties), so they stay
// consistent with the residual by construction.
//
// Like the CEED residual, interior edges are those owned by this rank; rows of ghost
// cells are kept in the COO pattern and PETSc sends them to their owners (the
// residual does the same via DMLocalToGlobal with ADD_VALUES).

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <petscdmceed.h>
#include <private/rdycoreimpl.h>
#include <private/rdyoperatorimpl.h>
#include <private/rdysweimpl.h>

#include "swe/swe_jacobian_ceed.h"

static inline CeedMemType MemTypeP2C(PetscMemType mem_type) { return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST; }

// fetches a residual sub-operator field's restriction and (passive) vector
static PetscErrorCode GetField(CeedOperator op, const char *name, CeedElemRestriction *rstr, CeedVector *vec) {
  PetscFunctionBeginUser;
  CeedOperatorField field;
  PetscCallCEED(CeedOperatorGetFieldByName(op, name, &field));
  PetscCallCEED(CeedOperatorFieldGetElemRestriction(field, rstr));
  if (vec) PetscCallCEED(CeedOperatorFieldGetVector(field, vec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// creates a restriction mapping element e (with ncomp values) of a sub-operator to
// positions [base + ncomp*e, base + ncomp*(e+1)) of the COO-values vector
static PetscErrorCode CreateCOORestriction(Ceed ceed, CeedInt num_elem, CeedInt ncomp, CeedSize base, CeedSize num_coo, CeedElemRestriction *rstr) {
  PetscFunctionBeginUser;
  CeedInt *offsets;
  PetscCall(PetscMalloc1(PetscMax(num_elem, 1), &offsets));
  for (CeedInt e = 0; e < num_elem; e++) offsets[e] = (CeedInt)(base + (CeedSize)ncomp * e);
  PetscCallCEED(CeedElemRestrictionCreate(ceed, num_elem, 1, ncomp, 1, num_coo, CEED_MEM_HOST, CEED_COPY_VALUES, offsets, rstr));
  PetscCall(PetscFree(offsets));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// appends the (row, col) indices of a 3x3 block for local cells (rc, cc)
static void AppendBlock(PetscInt *rows, PetscInt *cols, PetscInt *n, CeedInt row_offset, CeedInt col_offset) {
  for (PetscInt r = 0; r < 3; r++) {
    for (PetscInt c = 0; c < 3; c++) {
      rows[*n] = row_offset + r;
      cols[*n] = col_offset + c;
      (*n)++;
    }
  }
}

static PetscErrorCode CreateJacobianQFunction(Ceed ceed, const RDyConfig *config, CeedQFunctionUser f, const char *f_loc, CeedQFunction *qf) {
  PetscFunctionBeginUser;
  CeedQFunctionContext ctx;
  PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, f, f_loc, qf));
  PetscCall(CreateSWEQFunctionContext(ceed, *config, &ctx));
  PetscCallCEED(CeedQFunctionSetContext(*qf, ctx));
  PetscCallCEED(CeedQFunctionContextDestroy(&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Creates the composite Jacobian operator and the COO pattern (local dof indices).
static PetscErrorCode CreateCeedJacobianOperator(Operator *op) {
  PetscFunctionBeginUser;

  Ceed           ceed     = CeedContext();
  const PetscInt num_comp = op->num_components;
  PetscCheck(num_comp == 3, PETSC_COMM_SELF, PETSC_ERR_SUP, "The CEED analytic Jacobian supports SWE without tracers only");
  PetscCheck(!op->ceed.use_slope_reconstruction, PETSC_COMM_SELF, PETSC_ERR_SUP, "The CEED analytic Jacobian supports first-order fluxes only");

  CeedOperator *flux_subs, *source_subs;
  PetscCallCEED(CeedOperatorCompositeGetSubList(op->ceed.flux, &flux_subs));
  PetscCallCEED(CeedOperatorCompositeGetSubList(op->ceed.source, &source_subs));
  CeedOperator interior = flux_subs[0], source = source_subs[0];

  // count elements and COO entries
  CeedElemRestriction rstr_qL, rstr_qR, rstr_q;
  CeedInt             num_iedges, num_cells;
  PetscCall(GetField(interior, "q_left", &rstr_qL, NULL));
  PetscCall(GetField(interior, "q_right", &rstr_qR, NULL));
  PetscCall(GetField(source, "q", &rstr_q, NULL));
  PetscCallCEED(CeedElemRestrictionGetNumElements(rstr_qL, &num_iedges));
  PetscCallCEED(CeedElemRestrictionGetNumElements(rstr_q, &num_cells));

  CeedInt *num_bedges;
  PetscCall(PetscCalloc1(op->num_boundaries + 1, &num_bedges));
  CeedSize num_coo = (CeedSize)SWE_JAC_INTERIOR_NCOMP * num_iedges + (CeedSize)SWE_JAC_BLOCK_NCOMP * num_cells;
  for (PetscInt b = 0; b < op->num_boundaries; b++) {
    CeedElemRestriction rstr_b;
    PetscCall(GetField(flux_subs[1 + b], "q_left", &rstr_b, NULL));
    PetscCallCEED(CeedElemRestrictionGetNumElements(rstr_b, &num_bedges[b]));
    PetscCallCEED(CeedElemRestrictionDestroy(&rstr_b));
    num_coo += (CeedSize)SWE_JAC_BLOCK_NCOMP * num_bedges[b];
  }

  // COO pattern in local dof indices, in the same order as the values written below
  PetscInt *rows, *cols, n = 0;
  PetscCall(PetscMalloc1(num_coo, &rows));
  PetscCall(PetscMalloc1(num_coo, &cols));

  PetscCallCEED(CeedOperatorCreateComposite(ceed, &op->ceed.jacobian));
  PetscCallCEED(CeedVectorCreate(ceed, num_coo, &op->ceed.jac_coo_values));
  {
    CeedSize lsize;
    PetscCallCEED(CeedElemRestrictionGetLVectorSize(rstr_qL, &lsize));
    PetscCallCEED(CeedVectorCreate(ceed, lsize, &op->ceed.jac_state));
  }

  CeedSize base = 0;

  // interior edges
  {
    const CeedInt *offL, *offR;
    PetscCallCEED(CeedElemRestrictionGetOffsets(rstr_qL, CEED_MEM_HOST, &offL));
    PetscCallCEED(CeedElemRestrictionGetOffsets(rstr_qR, CEED_MEM_HOST, &offR));
    for (CeedInt e = 0; e < num_iedges; e++) {
      AppendBlock(rows, cols, &n, offL[e], offL[e]);
      AppendBlock(rows, cols, &n, offL[e], offR[e]);
      AppendBlock(rows, cols, &n, offR[e], offL[e]);
      AppendBlock(rows, cols, &n, offR[e], offR[e]);
    }
    PetscCallCEED(CeedElemRestrictionRestoreOffsets(rstr_qL, &offL));
    PetscCallCEED(CeedElemRestrictionRestoreOffsets(rstr_qR, &offR));

    CeedQFunction       qf;
    CeedOperator        sub;
    CeedElemRestriction rstr_geom, rstr_out;
    CeedVector          geom;
    CeedInt             ncomp_geom;
    PetscCall(GetField(interior, "geom", &rstr_geom, &geom));
    PetscCallCEED(CeedElemRestrictionGetNumComponents(rstr_geom, &ncomp_geom));
    switch (op->config->physics.flow.well_balancing) {
      case WELL_BALANCING_HR:
        PetscCall(CreateJacobianQFunction(ceed, op->config, SWEFluxJacobian_HydroRecon, SWEFluxJacobian_HydroRecon_loc, &qf));
        break;
      case WELL_BALANCING_AUGMENTED_ROE:
        PetscCall(CreateJacobianQFunction(ceed, op->config, SWEFluxJacobian_AugmentedRoe, SWEFluxJacobian_AugmentedRoe_loc, &qf));
        break;
      case WELL_BALANCING_NONE:
        PetscCall(CreateJacobianQFunction(ceed, op->config, SWEFluxJacobian_Roe, SWEFluxJacobian_Roe_loc, &qf));
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "The analytic Jacobian does not support this well_balancing method");
    }
    PetscCallCEED(CeedQFunctionAddInput(qf, "geom", ncomp_geom, CEED_EVAL_NONE));
    PetscCallCEED(CeedQFunctionAddInput(qf, "q_left", num_comp, CEED_EVAL_NONE));
    PetscCallCEED(CeedQFunctionAddInput(qf, "q_right", num_comp, CEED_EVAL_NONE));
    PetscCallCEED(CeedQFunctionAddOutput(qf, "blocks", SWE_JAC_INTERIOR_NCOMP, CEED_EVAL_NONE));
    PetscCall(CreateCOORestriction(ceed, num_iedges, SWE_JAC_INTERIOR_NCOMP, base, num_coo, &rstr_out));
    PetscCallCEED(CeedOperatorCreate(ceed, qf, NULL, NULL, &sub));
    PetscCallCEED(CeedOperatorSetName(sub, "interior flux jacobian"));
    PetscCallCEED(CeedOperatorSetField(sub, "geom", rstr_geom, CEED_BASIS_NONE, geom));
    PetscCallCEED(CeedOperatorSetField(sub, "q_left", rstr_qL, CEED_BASIS_NONE, op->ceed.jac_state));
    PetscCallCEED(CeedOperatorSetField(sub, "q_right", rstr_qR, CEED_BASIS_NONE, op->ceed.jac_state));
    PetscCallCEED(CeedOperatorSetField(sub, "blocks", rstr_out, CEED_BASIS_NONE, op->ceed.jac_coo_values));
    PetscCallCEED(CeedOperatorCompositeAddSub(op->ceed.jacobian, sub));
    PetscCallCEED(CeedOperatorDestroy(&sub));
    PetscCallCEED(CeedQFunctionDestroy(&qf));
    PetscCallCEED(CeedElemRestrictionDestroy(&rstr_geom));
    PetscCallCEED(CeedElemRestrictionDestroy(&rstr_out));
    PetscCallCEED(CeedVectorDestroy(&geom));
    base += (CeedSize)SWE_JAC_INTERIOR_NCOMP * num_iedges;
  }

  // boundary edges (one sub-operator per boundary, as in the residual)
  for (PetscInt b = 0; b < op->num_boundaries; b++) {
    CeedOperator        rsub = flux_subs[1 + b];
    CeedElemRestriction rstr_b, rstr_geom, rstr_out, rstr_D = NULL;
    CeedVector          geom, qD = NULL;
    CeedInt             ncomp_geom;
    PetscCall(GetField(rsub, "q_left", &rstr_b, NULL));
    PetscCall(GetField(rsub, "geom", &rstr_geom, &geom));
    PetscCallCEED(CeedElemRestrictionGetNumComponents(rstr_geom, &ncomp_geom));

    const CeedInt *offL;
    PetscCallCEED(CeedElemRestrictionGetOffsets(rstr_b, CEED_MEM_HOST, &offL));
    for (CeedInt e = 0; e < num_bedges[b]; e++) AppendBlock(rows, cols, &n, offL[e], offL[e]);
    PetscCallCEED(CeedElemRestrictionRestoreOffsets(rstr_b, &offL));

    CeedQFunction qf;
    switch (op->boundary_conditions[b].flow->type) {
      case CONDITION_DIRICHLET:
        PetscCall(CreateJacobianQFunction(ceed, op->config, SWEBoundaryFluxJacobian_Dirichlet_Roe, SWEBoundaryFluxJacobian_Dirichlet_Roe_loc, &qf));
        PetscCall(GetField(rsub, "q_dirichlet", &rstr_D, &qD));
        break;
      case CONDITION_REFLECTING:
        PetscCall(CreateJacobianQFunction(ceed, op->config, SWEBoundaryFluxJacobian_Reflecting_Roe, SWEBoundaryFluxJacobian_Reflecting_Roe_loc, &qf));
        break;
      case CONDITION_CRITICAL_OUTFLOW:
        PetscCall(CreateJacobianQFunction(ceed, op->config, SWEBoundaryFluxJacobian_Outflow_Roe, SWEBoundaryFluxJacobian_Outflow_Roe_loc, &qf));
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition for the CEED analytic Jacobian");
    }
    PetscCallCEED(CeedQFunctionAddInput(qf, "geom", ncomp_geom, CEED_EVAL_NONE));
    PetscCallCEED(CeedQFunctionAddInput(qf, "q_left", num_comp, CEED_EVAL_NONE));
    if (qD) PetscCallCEED(CeedQFunctionAddInput(qf, "q_dirichlet", num_comp, CEED_EVAL_NONE));
    PetscCallCEED(CeedQFunctionAddOutput(qf, "block", SWE_JAC_BLOCK_NCOMP, CEED_EVAL_NONE));
    PetscCall(CreateCOORestriction(ceed, num_bedges[b], SWE_JAC_BLOCK_NCOMP, base, num_coo, &rstr_out));

    CeedOperator sub;
    PetscCallCEED(CeedOperatorCreate(ceed, qf, NULL, NULL, &sub));
    PetscCallCEED(CeedOperatorSetName(sub, "boundary flux jacobian"));
    PetscCallCEED(CeedOperatorSetField(sub, "geom", rstr_geom, CEED_BASIS_NONE, geom));
    PetscCallCEED(CeedOperatorSetField(sub, "q_left", rstr_b, CEED_BASIS_NONE, op->ceed.jac_state));
    if (qD) PetscCallCEED(CeedOperatorSetField(sub, "q_dirichlet", rstr_D, CEED_BASIS_NONE, qD));
    PetscCallCEED(CeedOperatorSetField(sub, "block", rstr_out, CEED_BASIS_NONE, op->ceed.jac_coo_values));
    PetscCallCEED(CeedOperatorCompositeAddSub(op->ceed.jacobian, sub));
    PetscCallCEED(CeedOperatorDestroy(&sub));
    PetscCallCEED(CeedQFunctionDestroy(&qf));
    PetscCallCEED(CeedElemRestrictionDestroy(&rstr_b));
    PetscCallCEED(CeedElemRestrictionDestroy(&rstr_geom));
    PetscCallCEED(CeedElemRestrictionDestroy(&rstr_out));
    PetscCallCEED(CeedVectorDestroy(&geom));
    if (qD) {
      PetscCallCEED(CeedElemRestrictionDestroy(&rstr_D));
      PetscCallCEED(CeedVectorDestroy(&qD));
    }
    base += (CeedSize)SWE_JAC_BLOCK_NCOMP * num_bedges[b];
  }

  // cell sources
  {
    const CeedInt *off;
    PetscCallCEED(CeedElemRestrictionGetOffsets(rstr_q, CEED_MEM_HOST, &off));
    for (CeedInt e = 0; e < num_cells; e++) AppendBlock(rows, cols, &n, off[e], off[e]);
    PetscCallCEED(CeedElemRestrictionRestoreOffsets(rstr_q, &off));

    CeedQFunction       qf;
    CeedOperator        sub;
    CeedElemRestriction rstr_geom, rstr_mat, rstr_out;
    CeedVector          geom, mat_props;
    CeedInt             ncomp_geom, ncomp_mat;
    PetscCall(GetField(source, "geom", &rstr_geom, &geom));
    PetscCall(GetField(source, "mat_props", &rstr_mat, &mat_props));
    PetscCallCEED(CeedElemRestrictionGetNumComponents(rstr_geom, &ncomp_geom));
    PetscCallCEED(CeedElemRestrictionGetNumComponents(rstr_mat, &ncomp_mat));
    // HR and augmented Roe carry the bed slope in the edge flux, not in the cell source
    if (UsesEdgeBedSource(op->config)) {
      PetscCall(
          CreateJacobianQFunction(ceed, op->config, SWESourceJacobian_FullyImplicitNoBedSlope, SWESourceJacobian_FullyImplicitNoBedSlope_loc, &qf));
    } else {
      PetscCall(CreateJacobianQFunction(ceed, op->config, SWESourceJacobian_FullyImplicit, SWESourceJacobian_FullyImplicit_loc, &qf));
    }
    PetscCallCEED(CeedQFunctionAddInput(qf, "geom", ncomp_geom, CEED_EVAL_NONE));
    PetscCallCEED(CeedQFunctionAddInput(qf, "mat_props", ncomp_mat, CEED_EVAL_NONE));
    PetscCallCEED(CeedQFunctionAddInput(qf, "q", num_comp, CEED_EVAL_NONE));
    PetscCallCEED(CeedQFunctionAddOutput(qf, "block", SWE_JAC_BLOCK_NCOMP, CEED_EVAL_NONE));
    PetscCall(CreateCOORestriction(ceed, num_cells, SWE_JAC_BLOCK_NCOMP, base, num_coo, &rstr_out));
    PetscCallCEED(CeedOperatorCreate(ceed, qf, NULL, NULL, &sub));
    PetscCallCEED(CeedOperatorSetName(sub, "source jacobian"));
    PetscCallCEED(CeedOperatorSetField(sub, "geom", rstr_geom, CEED_BASIS_NONE, geom));
    PetscCallCEED(CeedOperatorSetField(sub, "mat_props", rstr_mat, CEED_BASIS_NONE, mat_props));
    PetscCallCEED(CeedOperatorSetField(sub, "q", rstr_q, CEED_BASIS_NONE, op->ceed.jac_state));
    PetscCallCEED(CeedOperatorSetField(sub, "block", rstr_out, CEED_BASIS_NONE, op->ceed.jac_coo_values));
    PetscCallCEED(CeedOperatorCompositeAddSub(op->ceed.jacobian, sub));
    PetscCallCEED(CeedOperatorDestroy(&sub));
    PetscCallCEED(CeedQFunctionDestroy(&qf));
    PetscCallCEED(CeedElemRestrictionDestroy(&rstr_geom));
    PetscCallCEED(CeedElemRestrictionDestroy(&rstr_mat));
    PetscCallCEED(CeedElemRestrictionDestroy(&rstr_out));
    PetscCallCEED(CeedVectorDestroy(&geom));
    PetscCallCEED(CeedVectorDestroy(&mat_props));
    base += (CeedSize)SWE_JAC_BLOCK_NCOMP * num_cells;
  }
  PetscCheck(base == num_coo && n == num_coo, PETSC_COMM_SELF, PETSC_ERR_PLIB, "COO bookkeeping mismatch");

  op->ceed.jac_num_coo  = num_coo;
  op->ceed.jac_coo_rows = rows;
  op->ceed.jac_coo_cols = cols;

  PetscCallCEED(CeedElemRestrictionDestroy(&rstr_qL));
  PetscCallCEED(CeedElemRestrictionDestroy(&rstr_qR));
  PetscCallCEED(CeedElemRestrictionDestroy(&rstr_q));
  PetscCall(PetscFree(num_bedges));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Evaluates the analytic Jacobian with libCEED and sets its values into J (COO).
/// @param [in]    op       the operator
/// @param [in]    u_local  local solution vector (owned + ghost cells)
/// @param [inout] J        Jacobian matrix created by DMCreateMatrix
PetscErrorCode ComputeSWEJacobianCeed(Operator *op, Vec u_local, Mat J) {
  PetscFunctionBegin;

  if (!op->ceed.jacobian) PetscCall(CreateCeedJacobianOperator(op));

  // set the COO pattern once per matrix; a composed object marks the matrix (composed
  // *data* would be invalidated by every MatSetValuesCOO, which changes the Mat state)
  PetscObject tag;
  PetscCall(PetscObjectQuery((PetscObject)J, "RDyJacobianCOOPattern", &tag));
  if (!tag) {
    PetscInt *rows, *cols;  // MatSetPreallocationCOOLocal overwrites its index arrays
    PetscCall(PetscMalloc2(op->ceed.jac_num_coo, &rows, op->ceed.jac_num_coo, &cols));
    PetscCall(PetscArraycpy(rows, op->ceed.jac_coo_rows, op->ceed.jac_num_coo));
    PetscCall(PetscArraycpy(cols, op->ceed.jac_coo_cols, op->ceed.jac_num_coo));
    PetscCall(MatSetPreallocationCOOLocal(J, op->ceed.jac_num_coo, rows, cols));
    PetscCall(PetscFree2(rows, cols));
    PetscContainer c;
    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &c));
    PetscCall(PetscObjectCompose((PetscObject)J, "RDyJacobianCOOPattern", (PetscObject)c));
    PetscCall(PetscContainerDestroy(&c));
  }

  // evaluate the blocks in the memory space of the state vector
  PetscMemType       mem_type;
  const PetscScalar *u_ptr;
  PetscCall(VecGetArrayReadAndMemType(u_local, &u_ptr, &mem_type));
  PetscCallCEED(CeedVectorSetArray(op->ceed.jac_state, MemTypeP2C(mem_type), CEED_USE_POINTER, (CeedScalar *)u_ptr));
  PetscCallCEED(CeedVectorSetValue(op->ceed.jac_coo_values, 0.0));  // passive outputs accumulate
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCEED(CeedOperatorApply(op->ceed.jacobian, CEED_VECTOR_NONE, CEED_VECTOR_NONE, CEED_REQUEST_IMMEDIATE));
  PetscCall(PetscLogGpuTimeEnd());
  CeedScalar *dummy;
  PetscCallCEED(CeedVectorTakeArray(op->ceed.jac_state, MemTypeP2C(mem_type), &dummy));
  PetscCall(VecRestoreArrayReadAndMemType(u_local, &u_ptr));

  const CeedScalar *vals;
  PetscCallCEED(CeedVectorGetArrayRead(op->ceed.jac_coo_values, MemTypeP2C(mem_type), &vals));
  PetscCall(MatSetValuesCOO(J, vals, INSERT_VALUES));
  PetscCallCEED(CeedVectorRestoreArrayRead(op->ceed.jac_coo_values, &vals));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Destroys the CEED Jacobian data held by the operator (if any).
PetscErrorCode DestroyCeedJacobian(Operator *op) {
  PetscFunctionBegin;
  if (op->ceed.jacobian) PetscCallCEED(CeedOperatorDestroy(&op->ceed.jacobian));
  if (op->ceed.jac_state) PetscCallCEED(CeedVectorDestroy(&op->ceed.jac_state));
  if (op->ceed.jac_coo_values) PetscCallCEED(CeedVectorDestroy(&op->ceed.jac_coo_values));
  PetscCall(PetscFree(op->ceed.jac_coo_rows));
  PetscCall(PetscFree(op->ceed.jac_coo_cols));
  PetscFunctionReturn(PETSC_SUCCESS);
}
