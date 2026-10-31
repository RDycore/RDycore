#ifndef SWE_AUGMENTED_ROE_FLUX_CEED_H
#define SWE_AUGMENTED_ROE_FLUX_CEED_H

#include <ceed/types.h>

#include "swe_types_ceed.h"

// disable compiler warnings for implicitly-declared math functions known to
// the JIT compiler
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wimplicit-function-declaration"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-function-declaration"

// Augmented Roe solver with an edge-based bed-source term, ported from
// SERGHEI's Solver::roe (src/Solvers.h). The PETSc twin of this file is
// swe_augmented_roe_flux_petsc.h; the two must stay in sync (see
// src/tests/test_augmented_roe.c).
//
// Differences from SERGHEI:
//   * friction is not folded into the bed-source wave strength beta
//   * the dt-dependent wet-wet subcritical clamp is omitted (RDycore runs with a
//     fixed dt that is assumed to satisfy the CFL condition)
//   * divisions by lambda_1 and lambda_3 in the intermediate depths are guarded

// fluctuations with a magnitude below this value are zeroed
#define AUGMENTED_ROE_ROUNDOFF_CEED 1e-14
// relative (to c~) eigenvalue magnitude below which an intermediate state is
// treated as admissible
#define AUGMENTED_ROE_EIG_EPS_CEED 1e-12

/// Computes the augmented Roe fluctuations D- (into the left cell) and D+
/// (into the right cell) across an edge with unit normal (cn, sn) pointing
/// from left to right. The left and right states are given in terms of depth
/// h and velocities (u, v); the conserved momenta used are h*u and h*v.
/// At least one of hl, hr must be positive.
/// @param [in]  gravity gravitational acceleration
/// @param [in]  tol     depth below which a cell is considered dry
/// @param [in]  hl, ul, vl, zl left depth, velocities, and bed elevation
/// @param [in]  hr, ur, vr, zr right depth, velocities, and bed elevation
/// @param [in]  sn, cn  sine and cosine of the edge normal angle
/// @param [out] dm      fluctuation D- sent to the left cell
/// @param [out] dp      fluctuation D+ sent to the right cell
/// @param [out] amax    maximum wave speed |u~n| + c~
CEED_QFUNCTION_HELPER void SWEAugmentedRoeFluctuations(const CeedScalar gravity, const CeedScalar tol, const CeedScalar hl, const CeedScalar ul,
                                                       const CeedScalar vl, const CeedScalar zl, const CeedScalar hr, const CeedScalar ur,
                                                       const CeedScalar vr, const CeedScalar zr, const CeedScalar sn, const CeedScalar cn,
                                                       CeedScalar dm[3], CeedScalar dp[3], CeedScalar *amax) {
  const CeedScalar nx = cn, ny = sn;
  const CeedScalar sqrt_gravity = sqrt(gravity);

  // Roe averages
  const CeedScalar sl     = sqrt(hl);
  const CeedScalar sr     = sqrt(hr);
  const CeedScalar htilde = 0.5 * (hl + hr);
  const CeedScalar utilde = (sl * ul + sr * ur) / (sl + sr);
  const CeedScalar vtilde = (sl * vl + sr * vr) / (sl + sr);
  const CeedScalar ctilde = sqrt(gravity * htilde);
  const CeedScalar un     = utilde * nx + vtilde * ny;

  // eigenvalues
  CeedScalar lambda[3]  = {un - ctilde, un, un + ctilde};
  CeedScalar lambdaE[3] = {0.0, 0.0, 0.0};

  // Harten-Hyman entropy fix: split a transonic rarefaction wave into left-
  // and right-going parts
  const CeedScalar unl = ul * nx + vl * ny;
  const CeedScalar unr = ur * nx + vr * ny;
  {
    const CeedScalar e1 = unl - sqrt_gravity * sl;
    const CeedScalar e2 = unr - sqrt_gravity * sr;
    if (e1 < 0.0 && e2 > 0.0) {
      const CeedScalar lambda_new = e1 * (e2 - lambda[0]) / (e2 - e1);
      lambdaE[0]                  = lambda[0] - lambda_new;
      lambda[0]                   = lambda_new;
    }
  }
  {
    const CeedScalar e1 = unl + sqrt_gravity * sl;
    const CeedScalar e2 = unr + sqrt_gravity * sr;
    if (e1 < 0.0 && e2 > 0.0) {
      const CeedScalar lambda_new = e2 * (lambda[2] - e1) / (e2 - e1);
      lambdaE[2]                  = lambda[2] - lambda_new;
      lambda[2]                   = lambda_new;
    }
  }

  // right eigenvectors (row k is the k-th eigenvector)
  const CeedScalar eigv[3][3] = {
      {1.0, utilde - ctilde * nx, vtilde - ctilde * ny},
      {0.0, -ctilde * ny,         ctilde * nx         },
      {1.0, utilde + ctilde * nx, vtilde + ctilde * ny},
  };

  // wave strengths from jumps in the conserved variables
  const CeedScalar dh  = hr - hl;
  const CeedScalar dqx = hr * ur - hl * ul;
  const CeedScalar dqy = hr * vr - hl * vl;
  const CeedScalar dqn = (dqx * nx + dqy * ny) - un * dh;
  CeedScalar       alpha[3];
  alpha[0] = 0.5 * (dh - dqn / ctilde);
  alpha[1] = ((dqy - vtilde * dh) * nx - (dqx - utilde * dh) * ny) / ctilde;
  alpha[2] = 0.5 * (dh + dqn / ctilde);

  // bed-source wave strength, clipped where the water lies below a step
  const CeedScalar dz  = zr - zl;
  CeedScalar       dzp = dz, hp;
  if (dz >= 0.0) {
    hp = hl;
    if (zl + hl < zr) dzp = hl;
  } else {
    hp = hr;
    if (zr + hr < zl) dzp = -hr;
  }
  const CeedScalar beta_b  = 0.5 / ctilde * gravity * (hp - 0.5 * fabs(dzp)) * dzp;
  const CeedScalar beta[3] = {beta_b, 0.0, -beta_b};

  // intermediate depths, guarding against vanishing eigenvalues
  const CeedScalar hstar  = hl + alpha[0];
  const CeedScalar eig_lo = AUGMENTED_ROE_EIG_EPS_CEED * ctilde;
  int              l1_neg = (fabs(lambda[0]) > eig_lo) && (hstar - beta[0] / lambda[0] < -tol);
  int              l2_neg = (fabs(lambda[2]) > eig_lo) && (hstar + beta[2] / lambda[2] < -tol);

  for (CeedInt j = 0; j < 3; j++) dm[j] = dp[j] = 0.0;

  if (l2_neg && hr < tol) {
    // wet/dry front: all mass goes to the left, no momentum fluctuation
    dm[0] = (lambda[0] * alpha[0] - beta[0]) + (lambda[2] * alpha[2] - beta[2]);
  } else if (l1_neg && hl < tol) {
    // wet/dry front: all mass goes to the right, no momentum fluctuation
    dp[0] = (lambda[0] * alpha[0] - beta[0]) + (lambda[2] * alpha[2] - beta[2]);
  } else {
    // regular upwind contributions
    for (CeedInt k = 0; k < 3; k++) {
      const CeedScalar w = lambda[k] * alpha[k] - beta[k];
      if (lambda[k] > 0.0) {
        for (CeedInt j = 0; j < 3; j++) dp[j] += w * eigv[k][j];
      } else {
        for (CeedInt j = 0; j < 3; j++) dm[j] += w * eigv[k][j];
      }
    }
  }

  // entropy-fix contributions (carry no source term)
  for (CeedInt k = 0; k < 3; k++) {
    const CeedScalar w = lambdaE[k] * alpha[k];
    if (lambdaE[k] > 0.0) {
      for (CeedInt j = 0; j < 3; j++) dp[j] += w * eigv[k][j];
    } else {
      for (CeedInt j = 0; j < 3; j++) dm[j] += w * eigv[k][j];
    }
  }

  // remove round-off
  for (CeedInt j = 0; j < 3; j++) {
    if (fabs(dm[j]) < AUGMENTED_ROE_ROUNDOFF_CEED) dm[j] = 0.0;
    if (fabs(dp[j]) < AUGMENTED_ROE_ROUNDOFF_CEED) dp[j] = 0.0;
  }

  *amax = fabs(un) + ctilde;
}

/// Computes the augmented Roe fluxes seen by the left and right cells of an
/// edge: flux_L = F_n(U_L) + D- and flux_R = F_n(U_R) - D+. A cell's update is
/// -(L/A_L) flux_L for the left cell and +(L/A_R) flux_R for the right cell.
/// The mass components are identical (flux_R[0] is set to flux_L[0]), so mass
/// is conserved exactly; a mass flux below round-off is zeroed so that
/// round-off velocities next to a wall or shoreline don't drain dry cells.
/// Negative (round-off) depths are treated as zero. Velocities use the same ANUGA regularization as
/// SWERiemannFlux_Roe, and the momenta h*u, h*v formed from them are used
/// consistently in F_n and in the jumps. At least one of qL.h, qR.h must exceed
/// tiny_h.
CEED_QFUNCTION_HELPER void SWERiemannFlux_AugmentedRoe(const CeedScalar gravity, const CeedScalar tiny_h, const CeedScalar h_anuga, SWEState qL,
                                                       SWEState qR, CeedScalar zL, CeedScalar zR, CeedScalar sn, CeedScalar cn, CeedScalar flux_L[3],
                                                       CeedScalar flux_R[3], CeedScalar *amax) {
  const CeedScalar hl = fmax(qL.h, 0.0), hr = fmax(qR.h, 0.0);

  const CeedScalar denom_l = hl * hl + h_anuga * h_anuga;
  const CeedScalar denom_r = hr * hr + h_anuga * h_anuga;

  const CeedScalar ul = (hl > tiny_h) ? qL.hu * hl / denom_l : 0.0;
  const CeedScalar vl = (hl > tiny_h) ? qL.hv * hl / denom_l : 0.0;
  const CeedScalar ur = (hr > tiny_h) ? qR.hu * hr / denom_r : 0.0;
  const CeedScalar vr = (hr > tiny_h) ? qR.hv * hr / denom_r : 0.0;

  CeedScalar dm[3], dp[3];
  SWEAugmentedRoeFluctuations(gravity, tiny_h, hl, ul, vl, zL, hr, ur, vr, zR, sn, cn, dm, dp, amax);

  const CeedScalar unl = ul * cn + vl * sn;
  const CeedScalar unr = ur * cn + vr * sn;

  flux_L[0] = hl * unl + dm[0];
  if (fabs(flux_L[0]) < AUGMENTED_ROE_ROUNDOFF_CEED) flux_L[0] = 0.0;
  flux_L[1] = hl * ul * unl + 0.5 * gravity * hl * hl * cn + dm[1];
  flux_L[2] = hl * vl * unl + 0.5 * gravity * hl * hl * sn + dm[2];

  flux_R[0] = flux_L[0];
  flux_R[1] = hr * ur * unr + 0.5 * gravity * hr * hr * cn - dp[1];
  flux_R[2] = hr * vr * unr + 0.5 * gravity * hr * hr * sn - dp[2];
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
#endif  // SWE_AUGMENTED_ROE_FLUX_CEED_H
