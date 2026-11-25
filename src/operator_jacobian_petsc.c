// Host (CPU) assembly of the analytic Jacobian dF/dU of the SWE right-hand side F
// for a fully implicit flow TS (paper Section 7). The right-hand side is
//
//   F_i(U) = sum_edges w_ie F_perp(U_L, U_R) + S(U_i) + external sources,
//
// with edge weights w = -L/A_left (left cell) and +L/A_right (right cell). Each
// internal edge contributes four 3x3 blocks; each boundary edge contributes to the
// diagonal block of its interior cell, with the chain rule through U_R(U_L); each
// cell contributes its source Jacobian. Pointwise derivatives come from the
// shared headers swe_roe_flux_jacobian_ceed.h and swe_sources_jacobian_ceed.h.
//
// This assembly reads the solution and operator data on the host, so it serves the
// PETSc backend and the CEED backend on CPUs. Like the PETSc residual, every rank
// visits all of its internal edges (owned and ghost) and writes only the rows of its
// owned cells, so no off-process entries are generated.

#include <private/rdycoreimpl.h>
#include <private/rdyoperatorimpl.h>

#include "swe/swe_edge_flux_jacobian_ceed.h"
#include "swe/swe_sources_jacobian_ceed.h"
#include "swe/swe_types_petsc.h"

// adds w * block (3x3) at (row cell, col cell) using local dof indices
static PetscErrorCode AddBlock(Mat J, PetscInt row_cell, PetscInt col_cell, PetscReal w, PetscReal block[3][3]) {
  PetscFunctionBegin;
  PetscInt  rows[3], cols[3];
  PetscReal vals[9];
  for (PetscInt k = 0; k < 3; k++) {
    rows[k] = 3 * row_cell + k;
    cols[k] = 3 * col_cell + k;
  }
  for (PetscInt i = 0; i < 3; i++)
    for (PetscInt j = 0; j < 3; j++) vals[3 * i + j] = w * block[i][j];
  PetscCall(MatSetValuesLocal(J, 3, rows, 3, cols, vals, ADD_VALUES));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Assembles the analytic Jacobian of the first-order SWE right-hand side on the host (plain Roe,
/// hydrostatic reconstruction, or augmented Roe interior fluxes).
/// @param [in]    op       the operator (mesh, boundaries, conditions, material data)
/// @param [in]    u_local  local solution vector (owned + ghost cells)
/// @param [inout] J        Jacobian matrix created by DMCreateMatrix
PetscErrorCode ComputeSWEJacobianHost(Operator *op, Vec u_local, Mat J) {
  PetscFunctionBegin;

  RDyMesh        *mesh    = op->mesh;
  RDyCells       *cells   = &mesh->cells;
  RDyEdges       *edges   = &mesh->edges;
  const PetscReal tiny_h  = op->config->physics.flow.tiny_h;
  const PetscReal h_anuga = op->config->physics.flow.h_anuga_regular;
  const PetscReal gravity = GRAVITY;

  PetscCheck(op->num_components == 3, PETSC_COMM_SELF, PETSC_ERR_SUP, "The analytic Jacobian supports SWE without tracers only");
  const RDyWellBalanceMethod wb = op->config->physics.flow.well_balancing;
  PetscCheck(wb == WELL_BALANCING_NONE || wb == WELL_BALANCING_HR || wb == WELL_BALANCING_AUGMENTED_ROE, PETSC_COMM_SELF, PETSC_ERR_SUP,
             "The analytic Jacobian supports well_balancing none, hydrostatic_reconstruction and augmented_roe");

  // cell bed elevations used by HR and augmented Roe (same as their residual operators)
  PetscReal *zc;
  PetscCall(PetscCalloc1(mesh->num_cells, &zc));
  if (UsesEdgeBedSource(op->config)) {
    PetscCall(RDyMeshGetCellBedElevations(mesh, op->config->grid.cell_elevation.file[0] ? PETSC_TRUE : PETSC_FALSE, zc));
  }

  PetscCall(MatZeroEntries(J));

  const PetscScalar *u_ptr;
  PetscCall(VecGetArrayRead(u_local, &u_ptr));

  // internal edges
  for (PetscInt e = 0; e < mesh->num_internal_edges; e++) {
    PetscInt edge_id = edges->internal_edge_ids[e];
    PetscInt l       = edges->cell_ids[2 * edge_id];
    PetscInt r       = edges->cell_ids[2 * edge_id + 1];
    if (r == -1) continue;
    PetscBool own_l = cells->is_owned[l], own_r = cells->is_owned[r];
    if (!own_l && !own_r) continue;

    SWEState qL = {u_ptr[3 * l], u_ptr[3 * l + 1], u_ptr[3 * l + 2]};
    SWEState qR = {u_ptr[3 * r], u_ptr[3 * r + 1], u_ptr[3 * r + 2]};
    if (qL.h < tiny_h && qR.h < tiny_h) continue;  // both dry: no flux

    // left/right fluxes G_L, G_R and blocks J = [dG_L/dU_L, dG_L/dU_R, dG_R/dU_L, dG_R/dU_R]
    PetscReal GL[3], GR[3], Jb[4][3][3], sn = edges->sn[edge_id], cn = edges->cn[edge_id];
    switch (op->config->physics.flow.well_balancing) {
      case WELL_BALANCING_HR:
        SWEEdgeFluxJacobian_HydroRecon(gravity, tiny_h, h_anuga, qL, qR, zc[l], zc[r], sn, cn, CeedEnabled() ? 0 : 1, GL, GR, Jb);
        break;
      case WELL_BALANCING_AUGMENTED_ROE:
        SWEEdgeFluxJacobian_AugmentedRoe(gravity, tiny_h, h_anuga, qL, qR, zc[l], zc[r], sn, cn, GL, GR, Jb);
        break;
      default:
        SWEEdgeFluxJacobian_Roe(gravity, tiny_h, h_anuga, qL, qR, sn, cn, GL, GR, Jb);
        break;
    }

    PetscReal len = edges->lengths[edge_id];
    if (own_l) {
      PetscReal w = -len / cells->areas[l];
      PetscCall(AddBlock(J, l, l, w, Jb[0]));
      PetscCall(AddBlock(J, l, r, w, Jb[1]));
    }
    if (own_r) {
      PetscReal w = len / cells->areas[r];
      PetscCall(AddBlock(J, r, l, w, Jb[2]));
      PetscCall(AddBlock(J, r, r, w, Jb[3]));
    }
  }
  PetscCall(PetscFree(zc));

  // boundary edges
  for (PetscInt b = 0; b < op->num_boundaries; b++) {
    RDyBoundary  boundary = op->boundaries[b];
    RDyCondition bc       = op->boundary_conditions[b];

    PetscReal *bvals = NULL;
    if (bc.flow->type == CONDITION_DIRICHLET && boundary.num_edges > 0) {
      PetscCall(PetscMalloc1(3 * boundary.num_edges, &bvals));
      PetscCall(GetOperatorBoundaryValues(op, boundary, 0, 3, boundary.num_edges, bvals));
    }

    for (PetscInt e = 0; e < boundary.num_edges; e++) {
      PetscInt edge_id = boundary.edge_ids[e];
      PetscInt l       = edges->cell_ids[2 * edge_id];
      if (!cells->is_owned[l]) continue;
      PetscReal sn = edges->sn[edge_id], cn = edges->cn[edge_id];

      SWEState  qL = {u_ptr[3 * l], u_ptr[3 * l + 1], u_ptr[3 * l + 2]};
      SWEState  qR;
      PetscReal dqR_dqL[3][3] = {{0}};
      PetscBool active        = PETSC_FALSE;

      switch (bc.flow->type) {
        case CONDITION_DIRICHLET:
          qR     = (SWEState){bvals[3 * e], bvals[3 * e + 1], bvals[3 * e + 2]};
          active = (PetscBool)(qL.h >= tiny_h || qR.h >= tiny_h);
          break;
        case CONDITION_REFLECTING: {
          PetscReal dum1 = sn * sn - cn * cn, dum2 = 2.0 * sn * cn;
          qR = (SWEState){qL.h, qL.hu * dum1 - qL.hv * dum2, -qL.hu * dum2 - qL.hv * dum1};
          SWEReflectingStateJacobian(sn, cn, dqR_dqL);
          active = (PetscBool)(qL.h > tiny_h);
        } break;
        case CONDITION_CRITICAL_OUTFLOW: {
          PetscReal q  = qL.hu * cn + qL.hv * sn;
          PetscReal hR = PetscPowReal(q * q / gravity, 1.0 / 3.0);
          PetscReal s  = PetscSqrtReal(gravity * hR);
          qR           = (SWEState){hR, hR * s * cn, hR * s * sn};
          SWECriticalOutflowStateJacobian(gravity, qL, sn, cn, dqR_dqL);
          active = (PetscBool)((qL.h > tiny_h || qR.h > tiny_h) && q >= 0.0);
        } break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition for the analytic Jacobian");
      }
      if (!active) continue;

      PetscReal flux[3], dFL[3][3], dFR[3][3], dFtot[3][3], amax;
      SWERiemannFluxJacobian_Roe(gravity, tiny_h, h_anuga, qL, qR, sn, cn, flux, dFL, dFR, &amax);
      SWEBoundaryFluxTotalJacobian(dFL, dFR, dqR_dqL, dFtot);
      PetscCall(AddBlock(J, l, l, -edges->lengths[edge_id] / cells->areas[l], dFtot));
    }
    PetscCall(PetscFree(bvals));
  }

  // cell sources (bed slope + fully implicit friction)
  OperatorData mat_props;
  PetscCall(GetOperatorDomainMaterialProperties(op, &mat_props));
  for (PetscInt c = 0; c < mesh->num_cells; c++) {
    if (!cells->is_owned[c]) continue;
    PetscInt  owned_id = cells->local_to_owned[c];
    SWEState  q        = {u_ptr[3 * c], u_ptr[3 * c + 1], u_ptr[3 * c + 2]};
    PetscReal n        = mat_props.values[MATERIAL_PROPERTY_MANNINGS][owned_id];
    PetscReal dS[3][3];
    // HR and augmented Roe carry the bed slope in the edge flux, not in the cell source
    PetscReal zx = UsesEdgeBedSource(op->config) ? 0.0 : cells->dz_dx[c], zy = UsesEdgeBedSource(op->config) ? 0.0 : cells->dz_dy[c];
    SWESourceJacobian(gravity, tiny_h, h_anuga, q, zx, zy, n, dS);
    PetscCall(AddBlock(J, c, c, 1.0, dS));
  }
  PetscCall(RestoreOperatorDomainMaterialProperties(op, &mat_props));

  PetscCall(VecRestoreArrayRead(u_local, &u_ptr));

  PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));

  PetscFunctionReturn(PETSC_SUCCESS);
}
