#ifndef SWE_JACOBIAN_CEED_H
#define SWE_JACOBIAN_CEED_H

// libCEED Q-functions that evaluate the analytic Jacobian dF/dU of the SWE
// right-hand side as explicit 3x3 blocks. Each Q-function writes its blocks into
// a passive output field that a restriction maps into one contiguous COO-values
// vector; the matching COO (row, col) pattern is built once on the host from the
// mesh (see operator_jacobian_ceed.c), and the values are handed to
// MatSetValuesCOO in the memory space where they were computed (host or device).
//
// This avoids libCEED's operator assembly (CeedOperatorLinearAssemble), which
// requires a single active basis in some libCEED versions and would need distinct
// identity bases for the left/right states of an edge.
//
// Field order matches the residual Q-functions (swe_fluxes_ceed.h,
// swe_sources_ceed.h), so the Jacobian sub-operators reuse the residual's
// restrictions and passive vectors (geometry, Dirichlet values, material
// properties). Dry branches write zero blocks.
//
// Block layout (row-major 3x3 blocks, 9 values each):
//   interior edge : [J_LL, J_LR, J_RL, J_RR]  (36 values)
//   boundary edge : [J_LL]                    (9 values)
//   cell source   : [J_cc]                    (9 values)

#include "swe_edge_flux_jacobian_ceed.h"
#include "swe_sources_jacobian_ceed.h"

#pragma GCC diagnostic   push
#pragma GCC diagnostic   ignored "-Wimplicit-function-declaration"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-function-declaration"
#pragma GCC diagnostic   push
#pragma GCC diagnostic   ignored "-Wvla"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla"

#define SWE_JAC_INTERIOR_NCOMP 36
#define SWE_JAC_BLOCK_NCOMP 9

// interior-edge flux schemes supported by the Jacobian
typedef enum {
  SWE_EDGE_FLUX_ROE = 0,        // plain Roe (well_balancing: none)
  SWE_EDGE_FLUX_HYDRO_RECON,    // hydrostatic reconstruction
  SWE_EDGE_FLUX_AUGMENTED_ROE,  // augmented Roe
} SWEEdgeFluxType;

/// Interior edges: in[0] geom, in[1] q_L, in[2] q_R; out[0] blocks (36 = w_L dG_L/dU_L, w_L dG_L/dU_R,
/// w_R dG_R/dU_L, w_R dG_R/dU_R). geom components 0..3 are (sn, cn, w_L = -L/A_L, w_R = L/A_R) for
/// every scheme; HR and augmented Roe also read the cell bed elevations (z_L, z_R) at 4, 5
/// (HRInteriorGeomIndex). For plain Roe, 4 and 5 hold vertex elevations and are not used.
CEED_QFUNCTION_HELPER int SWEFluxJacobian_Edge(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[], SWEEdgeFluxType type) {
  const CeedScalar(*geom)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_L)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*q_R)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  CeedScalar(*blocks)[CEED_Q_VLA]     = (CeedScalar(*)[CEED_Q_VLA])out[0];
  const SWEContext context            = (SWEContext)ctx;

  for (CeedInt i = 0; i < Q; i++) {
    SWEState         qL = {q_L[0][i], q_L[1][i], q_L[2][i]};
    SWEState         qR = {q_R[0][i], q_R[1][i], q_R[2][i]};
    const CeedScalar sn = geom[0][i], cn = geom[1][i], wL = geom[2][i], wR = geom[3][i];
    CeedScalar       GL[3], GR[3], J[4][3][3];
    switch (type) {
      case SWE_EDGE_FLUX_HYDRO_RECON:
        SWEEdgeFluxJacobian_HydroRecon(context->gravity, context->tiny_h, context->h_anuga_regular, qL, qR, geom[4][i], geom[5][i], sn, cn, GL, GR,
                                       J);
        break;
      case SWE_EDGE_FLUX_AUGMENTED_ROE:
        SWEEdgeFluxJacobian_AugmentedRoe(context->gravity, context->tiny_h, context->h_anuga_regular, qL, qR, geom[4][i], geom[5][i], sn, cn, GL, GR,
                                         J);
        break;
      default:
        SWEEdgeFluxJacobian_Roe(context->gravity, context->tiny_h, context->h_anuga_regular, qL, qR, sn, cn, GL, GR, J);
        break;
    }
    for (CeedInt r = 0; r < 3; r++) {
      for (CeedInt c = 0; c < 3; c++) {
        blocks[0 + 3 * r + c][i]  = wL * J[0][r][c];
        blocks[9 + 3 * r + c][i]  = wL * J[1][r][c];
        blocks[18 + 3 * r + c][i] = wR * J[2][r][c];
        blocks[27 + 3 * r + c][i] = wR * J[3][r][c];
      }
    }
  }
  return 0;
}

CEED_QFUNCTION(SWEFluxJacobian_Roe)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  return SWEFluxJacobian_Edge(ctx, Q, in, out, SWE_EDGE_FLUX_ROE);
}

CEED_QFUNCTION(SWEFluxJacobian_HydroRecon)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  return SWEFluxJacobian_Edge(ctx, Q, in, out, SWE_EDGE_FLUX_HYDRO_RECON);
}

CEED_QFUNCTION(SWEFluxJacobian_AugmentedRoe)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  return SWEFluxJacobian_Edge(ctx, Q, in, out, SWE_EDGE_FLUX_AUGMENTED_ROE);
}

/// Dirichlet boundary edges: in[0] geom (sn, cn, w_L, z_beg, z_end), in[1] q_L, in[2] q_dirichlet;
/// out[0] block J_LL (9)
CEED_QFUNCTION(SWEBoundaryFluxJacobian_Dirichlet_Roe)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const CeedScalar(*geom)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_L)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*q_D)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  CeedScalar(*block)[CEED_Q_VLA]      = (CeedScalar(*)[CEED_Q_VLA])out[0];
  const SWEContext context            = (SWEContext)ctx;

  for (CeedInt i = 0; i < Q; i++) {
    SWEState qL = {q_L[0][i], q_L[1][i], q_L[2][i]};
    SWEState qR = {q_D[0][i], q_D[1][i], q_D[2][i]};
    if (qL.h > context->tiny_h || qR.h > context->tiny_h) {
      CeedScalar flux[3], dFL[3][3], dFR[3][3], amax;
      SWERiemannFluxJacobian_Roe(context->gravity, context->tiny_h, context->h_anuga_regular, qL, qR, geom[0][i], geom[1][i], flux, dFL, dFR, &amax);
      for (CeedInt r = 0; r < 3; r++)
        for (CeedInt c = 0; c < 3; c++) block[3 * r + c][i] = geom[2][i] * dFL[r][c];
    } else {
      for (CeedInt k = 0; k < SWE_JAC_BLOCK_NCOMP; k++) block[k][i] = 0.0;
    }
  }
  return 0;
}

// shared body of the reflecting and critical-outflow boundary Jacobians
CEED_QFUNCTION_HELPER int SWEBoundaryFluxJacobian_Mapped(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[],
                                                         CeedInt is_outflow) {
  const CeedScalar(*geom)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_L)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  CeedScalar(*block)[CEED_Q_VLA]      = (CeedScalar(*)[CEED_Q_VLA])out[0];
  const SWEContext context            = (SWEContext)ctx;
  const CeedScalar gravity            = context->gravity;

  for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar sn = geom[0][i], cn = geom[1][i];
    SWEState         qL = {q_L[0][i], q_L[1][i], q_L[2][i]};
    SWEState         qR;
    CeedScalar       M[3][3];
    CeedInt          active;
    if (is_outflow) {
      const CeedScalar q  = qL.hu * cn + qL.hv * sn;
      const CeedScalar hR = pow(q * q / gravity, 1.0 / 3.0);
      const CeedScalar s  = sqrt(gravity * hR);
      qR.h                = hR;
      qR.hu               = hR * s * cn;
      qR.hv               = hR * s * sn;
      SWECriticalOutflowStateJacobian(gravity, qL, sn, cn, M);
      active = (qL.h > context->tiny_h || qR.h > context->tiny_h) && (q >= 0.0);
    } else {
      const CeedScalar dum1 = sn * sn - cn * cn, dum2 = 2.0 * sn * cn;
      qR.h  = qL.h;
      qR.hu = qL.hu * dum1 - qL.hv * dum2;
      qR.hv = -qL.hu * dum2 - qL.hv * dum1;
      SWEReflectingStateJacobian(sn, cn, M);
      active = (qL.h > context->tiny_h);
    }
    if (active) {
      CeedScalar flux[3], dFL[3][3], dFR[3][3], dF[3][3], amax;
      SWERiemannFluxJacobian_Roe(gravity, context->tiny_h, context->h_anuga_regular, qL, qR, sn, cn, flux, dFL, dFR, &amax);
      SWEBoundaryFluxTotalJacobian(dFL, dFR, M, dF);
      for (CeedInt r = 0; r < 3; r++)
        for (CeedInt c = 0; c < 3; c++) block[3 * r + c][i] = geom[2][i] * dF[r][c];
    } else {
      for (CeedInt k = 0; k < SWE_JAC_BLOCK_NCOMP; k++) block[k][i] = 0.0;
    }
  }
  return 0;
}

/// Reflecting boundary edges: in[0] geom, in[1] q_L; out[0] block J_LL (9)
CEED_QFUNCTION(SWEBoundaryFluxJacobian_Reflecting_Roe)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  return SWEBoundaryFluxJacobian_Mapped(ctx, Q, in, out, 0);
}

/// Critical-outflow boundary edges: in[0] geom, in[1] q_L; out[0] block J_LL (9)
CEED_QFUNCTION(SWEBoundaryFluxJacobian_Outflow_Roe)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  return SWEBoundaryFluxJacobian_Mapped(ctx, Q, in, out, 1);
}

/// Cell sources: in[0] geom (dz/dx, dz/dy), in[1] mat_props, in[2] q; out[0] block dS/dq (9).
/// With include_bed_slope = 0 (HR, augmented Roe) the bed slope lives in the edge flux.
CEED_QFUNCTION_HELPER int SWESourceJacobian_Cells(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[],
                                                  CeedInt include_bed_slope) {
  const CeedScalar(*geom)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*mat_props)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*q)[CEED_Q_VLA]         = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  CeedScalar(*block)[CEED_Q_VLA]           = (CeedScalar(*)[CEED_Q_VLA])out[0];
  const SWEContext context                 = (SWEContext)ctx;

  for (CeedInt i = 0; i < Q; i++) {
    SWEState         state = {q[0][i], q[1][i], q[2][i]};
    const CeedScalar zx = include_bed_slope ? geom[0][i] : 0.0, zy = include_bed_slope ? geom[1][i] : 0.0;
    CeedScalar       dS[3][3];
    SWESourceJacobian(context->gravity, context->tiny_h, context->h_anuga_regular, state, zx, zy, mat_props[MATERIAL_PROPERTY_MANNINGS][i], dS);
    for (CeedInt r = 0; r < 3; r++)
      for (CeedInt c = 0; c < 3; c++) block[3 * r + c][i] = dS[r][c];
  }
  return 0;
}

CEED_QFUNCTION(SWESourceJacobian_FullyImplicit)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  return SWESourceJacobian_Cells(ctx, Q, in, out, 1);
}

CEED_QFUNCTION(SWESourceJacobian_FullyImplicitNoBedSlope)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  return SWESourceJacobian_Cells(ctx, Q, in, out, 0);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
#pragma GCC diagnostic   pop
#pragma clang diagnostic pop

#endif
