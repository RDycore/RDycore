#ifndef SWE_EDGE_FLUX_JACOBIAN_CEED_H
#define SWE_EDGE_FLUX_JACOBIAN_CEED_H

// Analytic Jacobians of the interior-edge fluxes of the three Roe-type flux
// schemes, written in a common two-flux form: an edge contributes
//
//   cell_L += w_L G_L(U_L, U_R),   cell_R += w_R G_R(U_L, U_R),
//
// with w_L = -L/A_L and w_R = L/A_R. Plain Roe has G_L = G_R = F; hydrostatic
// reconstruction (HR) adds a different pressure correction to each side; the
// augmented Roe solver has G_L = F_n(U_L) + D^- and G_R = F_n(U_R) - D^+. The
// four 3x3 Jacobian blocks of an edge are
//
//   J[0] = dG_L/dU_L,  J[1] = dG_L/dU_R,  J[2] = dG_R/dU_L,  J[3] = dG_R/dU_R.
//
// Each function is the exact derivative of the corresponding residual kernel
// as coded (forward-mode chain rule with SWEDual, derivative of the active
// branch at non-differentiable points). Shared by the PETSc and CEED backends;
// GPU-JIT safe.

#include "swe_augmented_roe_flux_ceed.h"
#include "swe_roe_flux_jacobian_ceed.h"

#pragma GCC diagnostic   push
#pragma GCC diagnostic   ignored "-Wimplicit-function-declaration"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-function-declaration"

// copies the value and the left/right derivative blocks of G (3 dual components)
CEED_QFUNCTION_HELPER void SWEDualToBlocks(const SWEDual G[3], CeedScalar g[3], CeedScalar dG_dqL[3][3], CeedScalar dG_dqR[3][3]) {
  for (CeedInt i = 0; i < 3; i++) {
    g[i] = G[i].v;
    for (CeedInt j = 0; j < 3; j++) {
      dG_dqL[i][j] = G[i].d[j];
      dG_dqR[i][j] = G[i].d[3 + j];
    }
  }
}

CEED_QFUNCTION_HELPER void SWEEdgeJacobianZero(CeedScalar GL[3], CeedScalar GR[3], CeedScalar J[4][3][3]) {
  for (CeedInt i = 0; i < 3; i++) {
    GL[i] = GR[i] = 0.0;
    for (CeedInt k = 0; k < 4; k++)
      for (CeedInt j = 0; j < 3; j++) J[k][i][j] = 0.0;
  }
}

/// Plain Roe edge (well_balancing: none): G_L = G_R = F_Roe(U_L, U_R).
/// Matches SWEFlux (swe_fluxes_ceed.h) with dhv = 0.
CEED_QFUNCTION_HELPER void SWEEdgeFluxJacobian_Roe(const CeedScalar gravity, const CeedScalar tiny_h, const CeedScalar h_anuga, SWEState qL,
                                                   SWEState qR, CeedScalar sn, CeedScalar cn, CeedScalar GL[3], CeedScalar GR[3],
                                                   CeedScalar J[4][3][3]) {
  SWEEdgeJacobianZero(GL, GR, J);
  if (!(qL.h > tiny_h || qR.h > tiny_h)) return;
  CeedScalar amax;
  SWERiemannFluxJacobian_Roe(gravity, tiny_h, h_anuga, qL, qR, sn, cn, GL, J[0], J[1], &amax);
  for (CeedInt i = 0; i < 3; i++) {
    GR[i] = GL[i];
    for (CeedInt j = 0; j < 3; j++) {
      J[2][i][j] = J[0][i][j];
      J[3][i][j] = J[1][i][j];
    }
  }
}

/// Hydrostatic reconstruction edge (well_balancing: hydrostatic_reconstruction), matching
/// SWEFlux_HydroRecon (swe_fluxes_hydro_recon_ceed.h):
///   h*_K = max(0, h_K + z_K - max(z_L, z_R)),  q*_K = (h*_K, h*_K u_K, h*_K v_K)   (K = L, R)
///   F    = F_Roe(q*_L, q*_R)
///   G_K  = F + (g/2)(h_K^2 - h*_K^2) (0, cn, sn)
CEED_QFUNCTION_HELPER void SWEEdgeFluxJacobian_HydroRecon(const CeedScalar gravity, const CeedScalar tiny_h, const CeedScalar h_anuga, SWEState qL,
                                                          SWEState qR, CeedScalar zL, CeedScalar zR, CeedScalar sn, CeedScalar cn, CeedScalar GL[3],
                                                          CeedScalar GR[3], CeedScalar J[4][3][3]) {
  SWEEdgeJacobianZero(GL, GR, J);
  if (!(qL.h > tiny_h || qR.h > tiny_h)) return;

  const SWEDual hl = SWEDualVar(qL.h, 0), hul = SWEDualVar(qL.hu, 1), hvl = SWEDualVar(qL.hv, 2);
  const SWEDual hr = SWEDualVar(qR.h, 3), hur = SWEDualVar(qR.hu, 4), hvr = SWEDualVar(qR.hv, 5);

  // reconstructed depths and velocity-preserving momenta
  const CeedScalar z_max  = fmax(zL, zR);
  const SWEDual    hL_rec = SWEDualMax0(SWEDualAdd(hl, SWEDualConst(zL - z_max)));
  const SWEDual    hR_rec = SWEDualMax0(SWEDualAdd(hr, SWEDualConst(zR - z_max)));
  const SWEDual    uL = SWEDualVelocity(hl, hul, tiny_h, h_anuga), vL = SWEDualVelocity(hl, hvl, tiny_h, h_anuga);
  const SWEDual    uR = SWEDualVelocity(hr, hur, tiny_h, h_anuga), vR = SWEDualVelocity(hr, hvr, tiny_h, h_anuga);

  SWEDual F[3] = {SWEDualConst(0.0), SWEDualConst(0.0), SWEDualConst(0.0)};
  if (hL_rec.v > tiny_h || hR_rec.v > tiny_h) {
    CeedScalar amax;
    SWEDualRoeFlux(gravity, tiny_h, h_anuga, hL_rec, SWEDualMul(hL_rec, uL), SWEDualMul(hL_rec, vL), hR_rec, SWEDualMul(hR_rec, uR),
                   SWEDualMul(hR_rec, vR), sn, cn, F, &amax);
  }

  // hydrostatic pressure corrections, one per side
  const SWEDual corr_L = SWEDualScale(0.5 * gravity, SWEDualSub(SWEDualMul(hl, hl), SWEDualMul(hL_rec, hL_rec)));
  const SWEDual corr_R = SWEDualScale(0.5 * gravity, SWEDualSub(SWEDualMul(hr, hr), SWEDualMul(hR_rec, hR_rec)));

  SWEDual G_L[3], G_R[3];
  G_L[0] = F[0];
  G_L[1] = SWEDualAdd(F[1], SWEDualScale(cn, corr_L));
  G_L[2] = SWEDualAdd(F[2], SWEDualScale(sn, corr_L));
  G_R[0] = F[0];
  G_R[1] = SWEDualAdd(F[1], SWEDualScale(cn, corr_R));
  G_R[2] = SWEDualAdd(F[2], SWEDualScale(sn, corr_R));
  SWEDualToBlocks(G_L, GL, J[0], J[1]);
  SWEDualToBlocks(G_R, GR, J[2], J[3]);
}

// zeroes a dual whose value is below the augmented Roe round-off threshold (the kernel
// sets such values to exactly zero, so their derivative is zero too)
CEED_QFUNCTION_HELPER SWEDual SWEDualRoundoff(SWEDual a) { return (fabs(a.v) < AUGMENTED_ROE_ROUNDOFF_CEED) ? SWEDualConst(0.0) : a; }

/// Augmented Roe edge (well_balancing: augmented_roe), matching SWERiemannFlux_AugmentedRoe and
/// SWEAugmentedRoeFluctuations (swe_augmented_roe_flux_ceed.h) line by line:
///   G_L = F_n(U_L) + D^-,  G_R = F_n(U_R) - D^+  (mass components identical)
///   D^- = sum_{lambda_k <= 0} (lambda_k alpha_k - beta_k) r_k + entropy-fix terms, D^+ likewise.
CEED_QFUNCTION_HELPER void SWEEdgeFluxJacobian_AugmentedRoe(const CeedScalar gravity, const CeedScalar tiny_h, const CeedScalar h_anuga, SWEState qL,
                                                            SWEState qR, CeedScalar zl, CeedScalar zr, CeedScalar sn, CeedScalar cn, CeedScalar GL[3],
                                                            CeedScalar GR[3], CeedScalar J[4][3][3]) {
  SWEEdgeJacobianZero(GL, GR, J);
  if (!(qL.h > tiny_h || qR.h > tiny_h)) return;

  const CeedScalar tol = tiny_h, nx = cn, ny = sn, sqrt_gravity = sqrt(gravity);

  // depths at or below tiny_h are exactly dry; ANUGA-regularized velocities
  const SWEDual hl = (qL.h > tiny_h) ? SWEDualVar(qL.h, 0) : SWEDualConst(0.0);
  const SWEDual hr = (qR.h > tiny_h) ? SWEDualVar(qR.h, 3) : SWEDualConst(0.0);
  const SWEDual ul = SWEDualVelocity(hl, SWEDualVar(qL.hu, 1), tiny_h, h_anuga), vl = SWEDualVelocity(hl, SWEDualVar(qL.hv, 2), tiny_h, h_anuga);
  const SWEDual ur = SWEDualVelocity(hr, SWEDualVar(qR.hu, 4), tiny_h, h_anuga), vr = SWEDualVelocity(hr, SWEDualVar(qR.hv, 5), tiny_h, h_anuga);

  // Roe averages
  const SWEDual sl     = SWEDualSqrt(hl);
  const SWEDual sr     = SWEDualSqrt(hr);
  const SWEDual htilde = SWEDualScale(0.5, SWEDualAdd(hl, hr));
  const SWEDual ssum   = SWEDualAdd(sl, sr);
  const SWEDual utilde = SWEDualDiv(SWEDualAdd(SWEDualMul(sl, ul), SWEDualMul(sr, ur)), ssum);
  const SWEDual vtilde = SWEDualDiv(SWEDualAdd(SWEDualMul(sl, vl), SWEDualMul(sr, vr)), ssum);
  const SWEDual ctilde = SWEDualSqrt(SWEDualScale(gravity, htilde));
  const SWEDual un     = SWEDualAdd(SWEDualScale(nx, utilde), SWEDualScale(ny, vtilde));

  // eigenvalues and Harten-Hyman entropy fix
  SWEDual       lambda[3]  = {SWEDualSub(un, ctilde), un, SWEDualAdd(un, ctilde)};
  SWEDual       lambdaE[3] = {SWEDualConst(0.0), SWEDualConst(0.0), SWEDualConst(0.0)};
  const SWEDual unl        = SWEDualAdd(SWEDualScale(nx, ul), SWEDualScale(ny, vl));
  const SWEDual unr        = SWEDualAdd(SWEDualScale(nx, ur), SWEDualScale(ny, vr));
  {
    const SWEDual e1 = SWEDualSub(unl, SWEDualScale(sqrt_gravity, sl));
    const SWEDual e2 = SWEDualSub(unr, SWEDualScale(sqrt_gravity, sr));
    if (e1.v < 0.0 && e2.v > 0.0) {
      const SWEDual lambda_new = SWEDualDiv(SWEDualMul(e1, SWEDualSub(e2, lambda[0])), SWEDualSub(e2, e1));
      lambdaE[0]               = SWEDualSub(lambda[0], lambda_new);
      lambda[0]                = lambda_new;
    }
  }
  {
    const SWEDual e1 = SWEDualAdd(unl, SWEDualScale(sqrt_gravity, sl));
    const SWEDual e2 = SWEDualAdd(unr, SWEDualScale(sqrt_gravity, sr));
    if (e1.v < 0.0 && e2.v > 0.0) {
      const SWEDual lambda_new = SWEDualDiv(SWEDualMul(e2, SWEDualSub(lambda[2], e1)), SWEDualSub(e2, e1));
      lambdaE[2]               = SWEDualSub(lambda[2], lambda_new);
      lambda[2]                = lambda_new;
    }
  }

  // right eigenvectors (row k is the k-th eigenvector)
  const SWEDual eigv[3][3] = {
      {SWEDualConst(1.0), SWEDualSub(utilde, SWEDualScale(nx,   ctilde)),        SWEDualSub(vtilde, SWEDualScale(ny, ctilde))},
      {SWEDualConst(0.0), SWEDualScale(-ny,  ctilde),           SWEDualScale(nx, ctilde)          },
      {SWEDualConst(1.0),  SWEDualAdd(utilde,           SWEDualScale(nx, ctilde)), SWEDualAdd(vtilde,          SWEDualScale(ny,                ctilde))       },
  };

  // wave strengths
  const SWEDual dh  = SWEDualSub(hr, hl);
  const SWEDual dqx = SWEDualSub(SWEDualMul(hr, ur), SWEDualMul(hl, ul));
  const SWEDual dqy = SWEDualSub(SWEDualMul(hr, vr), SWEDualMul(hl, vl));
  const SWEDual dqn = SWEDualSub(SWEDualAdd(SWEDualScale(nx, dqx), SWEDualScale(ny, dqy)), SWEDualMul(un, dh));
  SWEDual       alpha[3];
  alpha[0] = SWEDualScale(0.5, SWEDualSub(dh, SWEDualDiv(dqn, ctilde)));
  alpha[1] = SWEDualDiv(
      SWEDualSub(SWEDualScale(nx, SWEDualSub(dqy, SWEDualMul(vtilde, dh))), SWEDualScale(ny, SWEDualSub(dqx, SWEDualMul(utilde, dh)))), ctilde);
  alpha[2] = SWEDualScale(0.5, SWEDualAdd(dh, SWEDualDiv(dqn, ctilde)));

  // bed-source wave strength, clipped where the water lies below a step
  const CeedScalar dz  = zr - zl;
  SWEDual          dzp = SWEDualConst(dz), hp;
  if (dz >= 0.0) {
    hp = hl;
    if (zl + hl.v < zr) dzp = hl;
  } else {
    hp = hr;
    if (zr + hr.v < zl) dzp = SWEDualScale(-1.0, hr);
  }
  // beta_b = 0.5 / c~ * g * (hp - 0.5 |dzp|) * dzp
  const SWEDual beta_b  = SWEDualScale(0.5 * gravity, SWEDualDiv(SWEDualMul(SWEDualSub(hp, SWEDualScale(0.5, SWEDualAbs(dzp))), dzp), ctilde));
  const SWEDual beta[3] = {beta_b, SWEDualConst(0.0), SWEDualScale(-1.0, beta_b)};

  // intermediate depths, guarding against vanishing eigenvalues (branch selection on values)
  const CeedScalar hstar  = hl.v + alpha[0].v;
  const CeedScalar eig_lo = AUGMENTED_ROE_EIG_EPS_CEED * ctilde.v;
  int              l1_neg = (fabs(lambda[0].v) > eig_lo) && (hstar - beta[0].v / lambda[0].v < -tol);
  int              l2_neg = (fabs(lambda[2].v) > eig_lo) && (hstar + beta[2].v / lambda[2].v < -tol);

  SWEDual dm[3], dp[3];
  for (CeedInt j = 0; j < 3; j++) dm[j] = dp[j] = SWEDualConst(0.0);

  if (l2_neg && hr.v < tol) {
    dm[0] = SWEDualAdd(SWEDualSub(SWEDualMul(lambda[0], alpha[0]), beta[0]), SWEDualSub(SWEDualMul(lambda[2], alpha[2]), beta[2]));
  } else if (l1_neg && hl.v < tol) {
    dp[0] = SWEDualAdd(SWEDualSub(SWEDualMul(lambda[0], alpha[0]), beta[0]), SWEDualSub(SWEDualMul(lambda[2], alpha[2]), beta[2]));
  } else {
    for (CeedInt k = 0; k < 3; k++) {
      const SWEDual w = SWEDualSub(SWEDualMul(lambda[k], alpha[k]), beta[k]);
      if (lambda[k].v > 0.0) {
        for (CeedInt j = 0; j < 3; j++) dp[j] = SWEDualAdd(dp[j], SWEDualMul(w, eigv[k][j]));
      } else {
        for (CeedInt j = 0; j < 3; j++) dm[j] = SWEDualAdd(dm[j], SWEDualMul(w, eigv[k][j]));
      }
    }
  }

  // entropy-fix contributions (no source term)
  for (CeedInt k = 0; k < 3; k++) {
    const SWEDual w = SWEDualMul(lambdaE[k], alpha[k]);
    if (lambdaE[k].v > 0.0) {
      for (CeedInt j = 0; j < 3; j++) dp[j] = SWEDualAdd(dp[j], SWEDualMul(w, eigv[k][j]));
    } else {
      for (CeedInt j = 0; j < 3; j++) dm[j] = SWEDualAdd(dm[j], SWEDualMul(w, eigv[k][j]));
    }
  }
  for (CeedInt j = 0; j < 3; j++) {
    dm[j] = SWEDualRoundoff(dm[j]);
    dp[j] = SWEDualRoundoff(dp[j]);
  }

  // left and right fluxes
  const SWEDual pl = SWEDualScale(0.5 * gravity, SWEDualMul(hl, hl));
  const SWEDual pr = SWEDualScale(0.5 * gravity, SWEDualMul(hr, hr));
  SWEDual       G_L[3], G_R[3];
  G_L[0] = SWEDualRoundoff(SWEDualAdd(SWEDualMul(hl, unl), dm[0]));
  G_L[1] = SWEDualAdd(SWEDualAdd(SWEDualMul(SWEDualMul(hl, ul), unl), SWEDualScale(cn, pl)), dm[1]);
  G_L[2] = SWEDualAdd(SWEDualAdd(SWEDualMul(SWEDualMul(hl, vl), unl), SWEDualScale(sn, pl)), dm[2]);
  G_R[0] = G_L[0];
  G_R[1] = SWEDualSub(SWEDualAdd(SWEDualMul(SWEDualMul(hr, ur), unr), SWEDualScale(cn, pr)), dp[1]);
  G_R[2] = SWEDualSub(SWEDualAdd(SWEDualMul(SWEDualMul(hr, vr), unr), SWEDualScale(sn, pr)), dp[2]);
  SWEDualToBlocks(G_L, GL, J[0], J[1]);
  SWEDualToBlocks(G_R, GR, J[2], J[3]);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop

#endif
