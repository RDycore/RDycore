#ifndef SWE_AUGMENTED_ROE_FLUX_PETSC_H
#define SWE_AUGMENTED_ROE_FLUX_PETSC_H

#include "swe_types_petsc.h"

// silence unused function warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"

// Augmented Roe solver with an edge-based bed-source term, ported from
// SERGHEI's Solver::roe (src/Solvers.h). The CEED twin of this file is
// swe_augmented_roe_flux_ceed.h; the two must stay in sync (see
// src/tests/test_augmented_roe.c).
//
// Differences from SERGHEI:
//   * friction is not folded into the bed-source wave strength beta
//   * the dt-dependent wet-wet subcritical clamp is omitted (RDycore runs with a
//     fixed dt that is assumed to satisfy the CFL condition)
//   * divisions by lambda_1 and lambda_3 in the intermediate depths are guarded

// fluctuations with a magnitude below this value are zeroed
#define AUGMENTED_ROE_ROUNDOFF 1e-14
// relative (to c~) eigenvalue magnitude below which an intermediate state is
// treated as admissible
#define AUGMENTED_ROE_EIG_EPS 1e-12

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
static void ComputeSWEAugmentedRoeFluctuationsPointwise(const PetscReal gravity, const PetscReal tol, const PetscReal hl, const PetscReal ul,
                                                        const PetscReal vl, const PetscReal zl, const PetscReal hr, const PetscReal ur,
                                                        const PetscReal vr, const PetscReal zr, const PetscReal sn, const PetscReal cn,
                                                        PetscReal dm[3], PetscReal dp[3], PetscReal *amax) {
  const PetscReal nx = cn, ny = sn;
  const PetscReal sqrt_gravity = PetscSqrtReal(gravity);

  // Roe averages
  const PetscReal sl     = PetscSqrtReal(hl);
  const PetscReal sr     = PetscSqrtReal(hr);
  const PetscReal htilde = 0.5 * (hl + hr);
  const PetscReal utilde = (sl * ul + sr * ur) / (sl + sr);
  const PetscReal vtilde = (sl * vl + sr * vr) / (sl + sr);
  const PetscReal ctilde = PetscSqrtReal(gravity * htilde);
  const PetscReal un     = utilde * nx + vtilde * ny;

  // eigenvalues
  PetscReal lambda[3]  = {un - ctilde, un, un + ctilde};
  PetscReal lambdaE[3] = {0.0, 0.0, 0.0};

  // Harten-Hyman entropy fix: split a transonic rarefaction wave into left-
  // and right-going parts
  const PetscReal unl = ul * nx + vl * ny;
  const PetscReal unr = ur * nx + vr * ny;
  {
    const PetscReal e1 = unl - sqrt_gravity * sl;
    const PetscReal e2 = unr - sqrt_gravity * sr;
    if (e1 < 0.0 && e2 > 0.0) {
      const PetscReal lambda_new = e1 * (e2 - lambda[0]) / (e2 - e1);
      lambdaE[0]                 = lambda[0] - lambda_new;
      lambda[0]                  = lambda_new;
    }
  }
  {
    const PetscReal e1 = unl + sqrt_gravity * sl;
    const PetscReal e2 = unr + sqrt_gravity * sr;
    if (e1 < 0.0 && e2 > 0.0) {
      const PetscReal lambda_new = e2 * (lambda[2] - e1) / (e2 - e1);
      lambdaE[2]                 = lambda[2] - lambda_new;
      lambda[2]                  = lambda_new;
    }
  }

  // right eigenvectors (row k is the k-th eigenvector)
  const PetscReal eigv[3][3] = {
      {1.0, utilde - ctilde * nx, vtilde - ctilde * ny},
      {0.0, -ctilde * ny,         ctilde * nx         },
      {1.0, utilde + ctilde * nx, vtilde + ctilde * ny},
  };

  // wave strengths from jumps in the conserved variables
  const PetscReal dh  = hr - hl;
  const PetscReal dqx = hr * ur - hl * ul;
  const PetscReal dqy = hr * vr - hl * vl;
  const PetscReal dqn = (dqx * nx + dqy * ny) - un * dh;
  PetscReal       alpha[3];
  alpha[0] = 0.5 * (dh - dqn / ctilde);
  alpha[1] = ((dqy - vtilde * dh) * nx - (dqx - utilde * dh) * ny) / ctilde;
  alpha[2] = 0.5 * (dh + dqn / ctilde);

  // bed-source wave strength, clipped where the water lies below a step
  const PetscReal dz  = zr - zl;
  PetscReal       dzp = dz, hp;
  if (dz >= 0.0) {
    hp = hl;
    if (zl + hl < zr) dzp = hl;
  } else {
    hp = hr;
    if (zr + hr < zl) dzp = -hr;
  }
  const PetscReal beta_b  = 0.5 / ctilde * gravity * (hp - 0.5 * PetscAbsReal(dzp)) * dzp;
  const PetscReal beta[3] = {beta_b, 0.0, -beta_b};

  // intermediate depths, guarding against vanishing eigenvalues
  const PetscReal hstar  = hl + alpha[0];
  const PetscReal eig_lo = AUGMENTED_ROE_EIG_EPS * ctilde;
  PetscBool       l1_neg = (PetscAbsReal(lambda[0]) > eig_lo) && (hstar - beta[0] / lambda[0] < -tol);
  PetscBool       l2_neg = (PetscAbsReal(lambda[2]) > eig_lo) && (hstar + beta[2] / lambda[2] < -tol);

  for (PetscInt j = 0; j < 3; j++) dm[j] = dp[j] = 0.0;

  if (l2_neg && hr < tol) {
    // wet/dry front: all mass goes to the left, no momentum fluctuation
    dm[0] = (lambda[0] * alpha[0] - beta[0]) + (lambda[2] * alpha[2] - beta[2]);
  } else if (l1_neg && hl < tol) {
    // wet/dry front: all mass goes to the right, no momentum fluctuation
    dp[0] = (lambda[0] * alpha[0] - beta[0]) + (lambda[2] * alpha[2] - beta[2]);
  } else {
    // regular upwind contributions
    for (PetscInt k = 0; k < 3; k++) {
      const PetscReal w = lambda[k] * alpha[k] - beta[k];
      if (lambda[k] > 0.0) {
        for (PetscInt j = 0; j < 3; j++) dp[j] += w * eigv[k][j];
      } else {
        for (PetscInt j = 0; j < 3; j++) dm[j] += w * eigv[k][j];
      }
    }
  }

  // entropy-fix contributions (carry no source term)
  for (PetscInt k = 0; k < 3; k++) {
    const PetscReal w = lambdaE[k] * alpha[k];
    if (lambdaE[k] > 0.0) {
      for (PetscInt j = 0; j < 3; j++) dp[j] += w * eigv[k][j];
    } else {
      for (PetscInt j = 0; j < 3; j++) dm[j] += w * eigv[k][j];
    }
  }

  // remove round-off
  for (PetscInt j = 0; j < 3; j++) {
    if (PetscAbsReal(dm[j]) < AUGMENTED_ROE_ROUNDOFF) dm[j] = 0.0;
    if (PetscAbsReal(dp[j]) < AUGMENTED_ROE_ROUNDOFF) dp[j] = 0.0;
  }

  *amax = PetscAbsReal(un) + ctilde;
}

/// Computes the augmented Roe fluxes seen by the left and right cells of an
/// edge: flux_l = F_n(U_L) + D- and flux_r = F_n(U_R) - D+. A cell's update is
/// -(L/A_L) flux_l for the left cell and +(L/A_R) flux_r for the right cell.
/// The mass components are identical (flux_r[0] is set to flux_l[0]), so mass
/// is conserved exactly. At least one of hl, hr must be positive.
static void ComputeSWEAugmentedRoeFluxPointwise(const PetscReal gravity, const PetscReal tol, const PetscReal hl, const PetscReal ul,
                                                const PetscReal vl, const PetscReal zl, const PetscReal hr, const PetscReal ur, const PetscReal vr,
                                                const PetscReal zr, const PetscReal sn, const PetscReal cn, PetscReal flux_l[3], PetscReal flux_r[3],
                                                PetscReal *amax) {
  PetscReal dm[3], dp[3];
  ComputeSWEAugmentedRoeFluctuationsPointwise(gravity, tol, hl, ul, vl, zl, hr, ur, vr, zr, sn, cn, dm, dp, amax);

  const PetscReal unl = ul * cn + vl * sn;
  const PetscReal unr = ur * cn + vr * sn;

  flux_l[0] = hl * unl + dm[0];
  flux_l[1] = hl * ul * unl + 0.5 * gravity * hl * hl * cn + dm[1];
  flux_l[2] = hl * vl * unl + 0.5 * gravity * hl * hl * sn + dm[2];

  flux_r[0] = flux_l[0];
  flux_r[1] = hr * ur * unr + 0.5 * gravity * hr * hr * cn - dp[1];
  flux_r[2] = hr * vr * unr + 0.5 * gravity * hr * hr * sn - dp[2];
}

/// Computes augmented Roe fluxes on a set of edges.
/// @param [in]  datal    left states (h, u, v) on the edges
/// @param [in]  datar    right states (h, u, v) on the edges
/// @param [in]  zl, zr   left and right cell bed elevations on the edges
/// @param [in]  sn, cn   sines and cosines of the edge normal angles
/// @param [in]  tiny_h   depth below which a cell is considered dry
/// @param [out] fluxes_l fluxes seen by the left cells (length 3*N)
/// @param [out] fluxes_r fluxes seen by the right cells (length 3*N)
/// @param [out] amax     maximum wave speeds on the edges (length N)
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode ComputeSWEAugmentedRoeFlux(RiemannStateData *datal, RiemannStateData *datar, const PetscReal *zl, const PetscReal *zr,
                                                 const PetscReal *sn, const PetscReal *cn, const PetscReal tiny_h, PetscReal *fluxes_l,
                                                 PetscReal *fluxes_r, PetscReal *amax) {
  PetscFunctionBeginUser;

  PetscAssert(datal->num_states == datar->num_states, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Size of data left and right of edges is not the same!");

  for (PetscInt i = 0; i < datal->num_states; ++i) {
    if (datal->h[i] > tiny_h || datar->h[i] > tiny_h) {
      ComputeSWEAugmentedRoeFluxPointwise(GRAVITY, tiny_h, datal->h[i], datal->u[i], datal->v[i], zl[i], datar->h[i], datar->u[i], datar->v[i], zr[i],
                                          sn[i], cn[i], &fluxes_l[3 * i], &fluxes_r[3 * i], &amax[i]);
    } else {
      for (PetscInt j = 0; j < 3; j++) fluxes_l[3 * i + j] = fluxes_r[3 * i + j] = 0.0;
      amax[i] = 0.0;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
#endif  // SWE_AUGMENTED_ROE_FLUX_PETSC_H
