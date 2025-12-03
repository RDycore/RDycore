#ifndef SWE_ROE_FLUX_PETSC_H
#define SWE_ROE_FLUX_PETSC_H

#include "swe_types_petsc.h"

// silence unused function warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"

// Computes eigenvalues lambda, right eigenvectors R, parameter change dW, and
// the maximum wave speed for the shallow water equations
static PetscErrorCode ComputeSWERoeEigenspectrum(const PetscReal hl, const PetscReal ul, const PetscReal vl, const PetscReal hr, const PetscReal ur,
                                                 const PetscReal vr, PetscReal sn, PetscReal cn, PetscReal lambda[3], PetscReal R[3][3],
                                                 PetscReal dW[3], PetscReal *amax) {
  PetscFunctionBeginUser;

  // compute Roe averages
  PetscReal duml  = pow(hl, 0.5);
  PetscReal dumr  = pow(hr, 0.5);
  PetscReal cl    = pow(GRAVITY * hl, 0.5);
  PetscReal cr    = pow(GRAVITY * hr, 0.5);
  PetscReal hhat  = duml * dumr;
  PetscReal uhat  = (duml * ul + dumr * ur) / (duml + dumr);
  PetscReal vhat  = (duml * vl + dumr * vr) / (duml + dumr);
  PetscReal chat  = pow(0.5 * GRAVITY * (hl + hr), 0.5);
  PetscReal uperp = uhat * cn + vhat * sn;

  PetscReal dh     = hr - hl;
  PetscReal du     = ur - ul;
  PetscReal dv     = vr - vl;
  PetscReal dupar  = -du * sn + dv * cn;
  PetscReal duperp = du * cn + dv * sn;

  // compute right eigenvectors
  R[0][0] = 1.0;
  R[0][1] = 0.0;
  R[0][2] = 1.0;
  R[1][0] = uhat - chat * cn;
  R[1][1] = -sn;
  R[1][2] = uhat + chat * cn;
  R[2][0] = vhat - chat * sn;
  R[2][1] = cn;
  R[2][2] = vhat + chat * sn;

  // compute eigenvalues
  PetscReal uperpl = ul * cn + vl * sn;
  PetscReal uperpr = ur * cn + vr * sn;
  PetscReal a1     = fabs(uperp - chat);
  PetscReal a2     = fabs(uperp);
  PetscReal a3     = fabs(uperp + chat);

  // apply critical flow fix
  PetscReal al1 = uperpl - cl;
  PetscReal ar1 = uperpr - cr;
  PetscReal da1 = fmax(0.0, 2.0 * (ar1 - al1));
  if (a1 < da1) {
    a1 = 0.5 * (a1 * a1 / da1 + da1);
  }
  PetscReal al3 = uperpl + cl;
  PetscReal ar3 = uperpr + cr;
  PetscReal da3 = fmax(0.0, 2.0 * (ar3 - al3));
  if (a3 < da3) {
    a3 = 0.5 * (a3 * a3 / da3 + da3);
  }
  lambda[0] = a1;
  lambda[1] = a2;
  lambda[2] = a3;

  // compute dW
  dW[0] = 0.5 * (dh - hhat * duperp / chat);
  dW[1] = hhat * dupar;
  dW[2] = 0.5 * (dh + hhat * duperp / chat);

  // max wave speed
  *amax = chat + fabs(uperp);

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Computes flux based on Roe solver
/// @param [in] *datal A RiemannDataSWE for values left of the edges
/// @param [in] *datar A RiemannDataSWE for values right of the edges
/// @param [in] sn array containing sines of the angles between edges and y-axis (length N)
/// @param [in] cn array containing cosines of the angles between edges and y-axis (length N)
/// @param [out] fij array containing fluxes through edges (length 3*N)
/// @param [out] amax array storing maximum courant number on edges (length N)
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode ComputeSWERoeFlux(RiemannStateData *datal, RiemannStateData *datar, const PetscReal *sn, const PetscReal *cn, PetscReal *fij,
                                        PetscReal *amax) {
  PetscFunctionBeginUser;

  PetscReal *hl = datal->h;
  PetscReal *ul = datal->u;
  PetscReal *vl = datal->v;

  PetscReal *hr = datar->h;
  PetscReal *ur = datar->u;
  PetscReal *vr = datar->v;

  PetscAssert(datal->num_states == datar->num_states, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Size of data left and right of edges is not the same!");

  PetscInt num_states = datal->num_states;
  for (PetscInt i = 0; i < num_states; ++i) {
    // compute eigenspectrum
    PetscReal A[3], R[3][3], dW[3];
    PetscCall(ComputeSWERoeEigenspectrum(hl[i], ul[i], vl[i], hr[i], ur[i], vr[i], sn[i], cn[i], A, R, dW, &amax[i]));

    // compute interface fluxes
    PetscReal uperpl = ul[i] * cn[i] + vl[i] * sn[i];
    PetscReal uperpr = ur[i] * cn[i] + vr[i] * sn[i];
    PetscReal FL[3]  = {
         uperpl * hl[i],
         ul[i] * uperpl * hl[i] + 0.5 * GRAVITY * hl[i] * hl[i] * cn[i],
         vl[i] * uperpl * hl[i] + 0.5 * GRAVITY * hl[i] * hl[i] * sn[i],
    };
    PetscReal FR[3] = {
        uperpr * hr[i],
        ur[i] * uperpr * hr[i] + 0.5 * GRAVITY * hr[i] * hr[i] * cn[i],
        vr[i] * uperpr * hr[i] + 0.5 * GRAVITY * hr[i] * hr[i] * sn[i],
    };

    // fij = 0.5*(FL + FR - matmul(R,matmul(A,dW))
    fij[3 * i + 0] = 0.5 * (FL[0] + FR[0] - R[0][0] * A[0] * dW[0] - R[0][1] * A[1] * dW[1] - R[0][2] * A[2] * dW[2]);
    fij[3 * i + 1] = 0.5 * (FL[1] + FR[1] - R[1][0] * A[0] * dW[0] - R[1][1] * A[1] * dW[1] - R[1][2] * A[2] * dW[2]);
    fij[3 * i + 2] = 0.5 * (FL[2] + FR[2] - R[2][0] * A[0] * dW[0] - R[2][1] * A[1] * dW[1] - R[2][2] * A[2] * dW[2]);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode JacobianOfScalarFlux(const PetscReal h, const PetscReal u, const PetscReal v, PetscReal sn, PetscReal cn, PetscReal J[3][3]) {
  PetscFunctionBeginUser;

  PetscReal q1 = h;
  PetscReal q2 = h * u;
  PetscReal q3 = h * v;

  PetscReal q1_pow2 = pow(q1, 2.0);

  J[0][0] = 0.0;
  J[0][1] = cn;
  J[0][2] = sn;

  J[1][0] = (-pow(q2, 2.0) * cn - q2 * q3 * sn ) / q1_pow2 +  GRAVITY * q1 * cn;
  J[1][1] = 2.0 * q2 * cn / q1 + q3 * sn / q1;
  J[1][2] = q2 * sn / q1;

  J[2][0] = (-q2 * q3 * cn - pow(q3, 2.0) * sn ) / q1_pow2 + GRAVITY * q1 * sn;
  J[2][1] = q3 * cn / q1;
  J[2][2] = q2 * cn / q1 + 2.0 * q3 * sn / q1;

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeLambdaAndDerivative (const PetscReal q1L, const PetscReal q2L, const PetscReal q3L,
                                           const PetscReal q1R, const PetscReal q2R, const PetscReal q3R,
                                           const PetscReal sn, const PetscReal cn,
                                           const PetscReal HatUperp, const PetscReal HatA,
                                           const PetscReal dHatUperp_dQl[3], const PetscReal dHatA_dQl[3],
                                           const PetscReal dHatUperp_dQr[3], const PetscReal dHatA_dQr[3],
                                           const PetscBool compute_Lambda1,
                                           PetscReal *LambdaStar,
                                           PetscReal dLambdaStar_dQl[3],
                                           PetscReal dLambdaStar_dQr[3]) {

  PetscFunctionBeginUser;

  PetscReal factor = (compute_Lambda1) ? -1 : 1;

  // compute lambda and its derivatives
  PetscReal UperpL = q2L / q1L * cn + q3L / q1L * sn;
  PetscReal UperpR = q2R / q1R * cn + q3R / q1R * sn;

  PetscReal LambdaL = UperpL + factor * pow(GRAVITY * q1L, 0.5);
  PetscReal LambdaR = UperpR + factor * pow(GRAVITY * q1R, 0.5);

  PetscReal dLambdaL_dQl[3] = { (-q2L * cn - q3L * sn) / (q1L * q1L) + factor * 0.5 * pow(GRAVITY / q1L, 0.5),
                                 cn / q1L,
                                 sn / q1L};
  PetscReal dLambdaR_dQr[3] = { (-q2R * cn - q3R * sn) / (q1R * q1R) + factor * 0.5 * pow(GRAVITY / q1R, 0.5),
                                 cn / q1R,
                                 sn / q1R};

  // compute delta lambda and its derivatives
  PetscReal DelLambda = 4.0 * (LambdaR - LambdaL);
  PetscReal dDelLambda_dQl[3] = { -4.0 * dLambdaL_dQl[0],
                                    -4.0 * dLambdaL_dQl[1],
                                    -4.0 * dLambdaL_dQl[2]};
  PetscReal dDelLambda_dQr[3] = { 4.0 * dLambdaR_dQr[0],
                                    4.0 * dLambdaR_dQr[1],
                                    4.0 * dLambdaR_dQr[2]};


  // compute HatLambda and its derivatives
  PetscReal HatLambda = (HatUperp - HatA);
  PetscReal dHatLambda_dQl[3] = { dHatUperp_dQl[0] - dHatA_dQl[0],
                                   dHatUperp_dQl[1] - dHatA_dQl[1],
                                   dHatUperp_dQl[2] - dHatA_dQl[2]};
  PetscReal dHatLambda_dQr[3] = { dHatUperp_dQr[0] - dHatA_dQr[0],
                                   dHatUperp_dQr[1] - dHatA_dQr[1],
                                   dHatUperp_dQr[2] - dHatA_dQr[2]};

  if (fabs(HatLambda) < 0.5 * fabs(DelLambda) ) {
    PetscReal factor = (HatLambda > 0.0) ? 1.0 : -1.0;

    // apply correction
    *LambdaStar = factor * (HatLambda * HatLambda) / DelLambda + DelLambda / 4.0;

    dLambdaStar_dQl[0] = factor * ( (2.0 * HatLambda * dHatLambda_dQl[0] * DelLambda - HatLambda * HatLambda * dDelLambda_dQl[0]) / (DelLambda * DelLambda) + 0.25 * dDelLambda_dQl[0]);
    dLambdaStar_dQl[1] = factor * ( (2.0 * HatLambda * dHatLambda_dQl[1] * DelLambda - HatLambda * HatLambda * dDelLambda_dQl[1]) / (DelLambda * DelLambda) + 0.25 * dDelLambda_dQl[1]);
    dLambdaStar_dQl[2] = factor * ( (2.0 * HatLambda * dHatLambda_dQl[2] * DelLambda - HatLambda * HatLambda * dDelLambda_dQl[2]) / (DelLambda * DelLambda) + 0.25 * dDelLambda_dQl[2]);

    dLambdaStar_dQr[0] = factor * ( (2.0 * HatLambda * dHatLambda_dQr[0] * DelLambda - HatLambda * HatLambda * dDelLambda_dQr[0]) / (DelLambda * DelLambda) + 0.25 * dDelLambda_dQr[0]);
    dLambdaStar_dQr[1] = factor * ( (2.0 * HatLambda * dHatLambda_dQr[1] * DelLambda - HatLambda * HatLambda * dDelLambda_dQr[1]) / (DelLambda * DelLambda) + 0.25 * dDelLambda_dQr[1]);
    dLambdaStar_dQr[2] = factor * ( (2.0 * HatLambda * dHatLambda_dQr[2] * DelLambda - HatLambda * HatLambda * dDelLambda_dQr[2]) / (DelLambda * DelLambda) + 0.25 * dDelLambda_dQr[2]);

  } else {

    PetscReal factor = (HatLambda > 0.0) ? 1.0 : -1.0;

    // no correction
    *LambdaStar = factor * HatLambda;

    dLambdaStar_dQl[0] = factor * dHatLambda_dQl[0];
    dLambdaStar_dQl[1] = factor * dHatLambda_dQl[1];
    dLambdaStar_dQl[2] = factor * dHatLambda_dQl[2];

    dLambdaStar_dQr[0] = factor * dHatLambda_dQr[0];
    dLambdaStar_dQr[1] = factor * dHatLambda_dQr[1];
    dLambdaStar_dQr[2] = factor * dHatLambda_dQr[2];
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode JacobianOfDissipiationTerm(const PetscReal hl, const PetscReal ul, const PetscReal vl, const PetscReal hr, const PetscReal ur,
                                                 const PetscReal vr, PetscReal sn, PetscReal cn, PetscReal Jl[3][3],PetscReal Jr[3][3]) {
  PetscFunctionBeginUser;

  PetscReal q1L = hl;
  PetscReal q2L = hl * ul;
  PetscReal q3L = hl * vl;

  PetscReal q1R = hr;
  PetscReal q2R = hr * ur;
  PetscReal q3R = hr * vr;

  PetscReal q1L_sqrt = pow(q1L, 0.5);
  PetscReal q1R_sqrt = pow(q1R, 0.5);

  // compute Roe averages: HatU and derivatives
  PetscReal HatU = ( q2L / q1L_sqrt + q2R / q1R_sqrt) / (q1L_sqrt + q1R_sqrt);
  PetscReal common_denom = 2.0 * q1L_sqrt * q1R_sqrt * pow(q1L_sqrt + q1R_sqrt, 2.0);
  PetscReal dHatU_dQl[3] = { - (q1L * q2R + 2.0 * q2L * q1L_sqrt * q1R_sqrt + q2L * q1R) / (q1L * common_denom),
                             1.0 / (q1L_sqrt * (q1L_sqrt + q1R_sqrt)),
                             0.0};
  PetscReal dHatU_dQr[3] = { - (q1R * q2L + 2.0 * q2R * q1L_sqrt * q1R_sqrt + q2R * q1L) / (q1R * common_denom),
                             1.0 / (q1R_sqrt * (q1L_sqrt + q1R_sqrt)),
                             0.0};

    // compute Roe averages: HatV and its derivatives
  PetscReal HatV = ( q3L / q1L_sqrt + q3R / q1R_sqrt) / (q1L_sqrt + q1R_sqrt);
  PetscReal dHatV_dQl[3] = { - (q1L * q3R + 2.0 * q3L * q1L_sqrt * q1R_sqrt + q3L * q1R) / (q1L * common_denom),
                             0.0,
                             1.0 / (q1L_sqrt * (q1L_sqrt + q1R_sqrt))};
  PetscReal dHatV_dQr[3] = { - (q1R * q3L + 2.0 * q3R * q1L_sqrt * q1R_sqrt + q3R * q1L) / (q1R * common_denom),
                             0.0,
                             1.0 / (q1R_sqrt * (q1L_sqrt + q1R_sqrt))};

  // compute HatA and its derivatives
  PetscReal HatA  = pow(0.5 * GRAVITY * (q1L + q1R), 0.5);
  PetscReal dHatA_dQl[3] = {0.25 * GRAVITY / HatA, 0.0, 0.0};
  PetscReal dHatA_dQr[3] = {0.25 * GRAVITY / HatA, 0.0, 0.0};


  // compute dUperp and its derivatives
  PetscReal dUperp = (q2R / q1R - q2L / q1L) * cn + (q3R / q1R - q3L / q1L) * sn;
  PetscReal ddUperp_dQl[3] = { (q2L * cn + q3L * sn) / (q1L * q1L),
                               - cn / q1L,
                               - sn / q1L};
  PetscReal ddUperp_dQr[3] = { -(q2R * cn + q3R * sn) / (q1R * q1R),
                               cn / q1R,
                               sn / q1R};

  // compute dupar and its derivatives
  PetscReal dupar = -(q2R / q1R - q2L / q1L) * sn + (q3R / q1R - q3L / q1L) * cn;
  PetscReal ddupar_dQl[3] = { -(q2L * sn + q3L * cn) / (q1L * q1L),
                               sn / q1L,
                               cn / q1L};
  PetscReal ddupar_dQr[3] = { (q2R * sn + q3R * cn) / (q1R * q1R),
                               -sn / q1R,
                               -cn / q1R};

  // compute HatUperp and its derivatives
  PetscReal HatUperp = HatU * cn + HatV * sn;
  PetscReal dHatUperp_dQl[3] = { dHatU_dQl[0] * cn + dHatV_dQl[0] * sn,
                                 dHatU_dQl[1] * cn + dHatV_dQl[1] * sn,
                                 dHatU_dQl[2] * cn + dHatV_dQl[2] * sn};
  PetscReal dHatUperp_dQr[3] = { dHatU_dQr[0] * cn + dHatV_dQr[0] * sn,
                                 dHatU_dQr[1] * cn + dHatV_dQr[1] * sn,
                                 dHatU_dQr[2] * cn + dHatV_dQr[2] * sn};

  // compute dV and its derivatives
  PetscReal dV[3], dV_dQl[3][3], dV_dQr[3][3];
  {
    // compute Roe averages: HatH and its derivatives
    PetscReal HatH  = pow(q1L * q1R, 0.5);
    PetscReal dHatH_dQrl[3] = {pow(q1R / q1L, 0.5) * 0.5, 0.0, 0.0};
    PetscReal dHatH_dQrr[3] = {pow(q1L / q1R, 0.5) * 0.5, 0.0, 0.0};

    PetscReal dh = q1R - q1L;
    PetscReal dh_dQl[3] = { -1.0, 0.0, 0.0};
    PetscReal dh_dQr[3] = { 1.0, 0.0, 0.0};

    dV[0] = 0.5 * (dh - HatH * dUperp / HatA);
    dV[1] = HatH * dupar;
    dV[2] = 0.5 * (dh + HatH * dUperp / HatA);

    for (PetscInt k = 0; k < 3; ++k) {
      dV_dQl[0][k] = 0.5 * (dh_dQl[k] - (dHatH_dQrl[k] * dUperp + HatH * ddUperp_dQl[k]) / HatA + HatH * dUperp * dHatA_dQl[k] / (HatA * HatA));
      dV_dQl[1][k] = dHatH_dQrl[k] * dupar + HatH * ddupar_dQl[k];
      dV_dQl[2][k] = 0.5 * (dh_dQl[k] + (dHatH_dQrl[k] * dUperp + HatH * ddUperp_dQl[k]) / HatA - HatH * dUperp * dHatA_dQl[k] / (HatA * HatA));

      dV_dQr[0][k] = 0.5 * (dh_dQr[k] - (dHatH_dQrr[k] * dUperp + HatH * ddUperp_dQr[k]) / HatA + HatH * dUperp * dHatA_dQr[k] / (HatA * HatA));
      dV_dQr[1][k] = dHatH_dQrr[k] * dupar + HatH * ddupar_dQr[k];
      dV_dQr[2][k] = 0.5 * (dh_dQr[k] + (dHatH_dQrr[k] * dUperp + HatH * ddUperp_dQr[k]) / HatA - HatH * dUperp * dHatA_dQr[k] / (HatA * HatA));
    }
  }

  // compute Lambda_star and its derivatives
  PetscBool compute_Lambda1;

  PetscReal Lambda1Star, dLambda1Star_dQl[3], dLambda1Star_dQr[3];
  compute_Lambda1 = PETSC_FALSE;
  PetscCall(ComputeLambdaAndDerivative (q1L, q2L, q3L,
                                q1R, q2R, q3R,
                                sn, cn,
                                HatUperp, HatA,
                                dHatUperp_dQl, dHatA_dQl,
                                dHatUperp_dQr, dHatA_dQr,
                                compute_Lambda1,
                                &Lambda1Star,
                                dLambda1Star_dQl, dLambda1Star_dQr));

  PetscReal Lambda3Star, dLambda3Star_dQl[3], dLambda3Star_dQr[3];
  compute_Lambda1 = PETSC_FALSE;
  PetscCall(ComputeLambdaAndDerivative (q1L, q2L, q3L,
                                q1R, q2R, q3R,
                                sn, cn,
                                HatUperp, HatA,
                                dHatUperp_dQl, dHatA_dQl,
                                dHatUperp_dQr, dHatA_dQr,
                                compute_Lambda1,
                                &Lambda3Star,
                                dLambda3Star_dQl, dLambda3Star_dQr));

  PetscReal Lambda2Star, dLambda2Star_dQl[3], dLambda2Star_dQr[3];
  PetscReal factor = (HatUperp > 0.0) ? 1.0 : -1.0;
  Lambda2Star = factor * HatUperp;
  dLambda2Star_dQl[0] = factor * dHatUperp_dQl[0];
  dLambda2Star_dQl[1] = factor * dHatUperp_dQl[1];
  dLambda2Star_dQl[2] = factor * dHatUperp_dQl[2];

  dLambda2Star_dQr[0] = factor * dHatUperp_dQr[0];
  dLambda2Star_dQr[1] = factor * dHatUperp_dQr[1];
  dLambda2Star_dQr[2] = factor * dHatUperp_dQr[2];

  PetscReal R[3][3] = {
    {1.0,          0.0,           1.0},
    {HatU - HatA * cn, -sn,        HatU + HatA * cn},
    {HatV - HatA * sn,  cn,        HatV + HatA * sn}
  };

  PetscReal GL[3][3][3], GR[3][3][3], HL[3][3][3], HR[3][3][3];

  // initialize derivatives of R to zero
  for (PetscInt i = 0; i < 3; ++i) {
    for (PetscInt j = 0; j < 3; ++j) {
      for (PetscInt k = 0; k < 3; ++k) {
        GL[i][j][k] = 0.0;
        GR[i][j][k] = 0.0;
        HL[i][j][k] = 0.0;
        HR[i][j][k] = 0.0;
      }
    }
  }

  // only fill diagonals since others will be multiplied by zero later
  for (PetscInt k = 0; k < 3; k++) {
    PetscReal dR_dQl = dHatV_dQl[k] + dHatA_dQl[k] * sn;
    PetscReal dR_dQr = dHatV_dQr[k] + dHatA_dQr[k] * sn;

    GL[2][2][k] = dR_dQl * Lambda3Star;
    GR[2][2][k] = dR_dQr * Lambda3Star;

    HL[0][0][k] = R[0][0] * dLambda1Star_dQl[k];
    HL[1][1][k] = R[1][1] * dLambda2Star_dQl[k];
    HL[2][2][k] = R[2][2] * dLambda3Star_dQl[k];
  }

  // initialize Jacobians to zero
  for (PetscInt i = 0; i < 3; ++i) {
    for (PetscInt j = 0; j < 3; ++j) {
      for (PetscInt k = 0; k < 3; ++k) {
        Jl[i][j] = 0.0;
        Jr[i][j] = 0.0;
      }
    }
  }

  for (PetscInt k = 0; k < 3; ++k) {
    for (PetscInt i = 0; i < 3; ++i) {
      PetscReal tmpL = 0.0, tmpR = 0.0;
      for (PetscInt j = 0; j < 3; ++j) {
        tmpL += GL[i][j][k] * dV[k];
        tmpR += GR[i][j][k] * dV[k];

        tmpL += HL[i][j][k] * dV[k];
        tmpR += HR[i][j][k] * dV[k];
      }
      Jl[i][k] += tmpL;
      Jr[i][k] += tmpR;
    }
  }

  PetscReal RtimesL[3][3];
  for (PetscInt i = 0; i < 3; ++i) {
    RtimesL[i][0] = R[i][0] * Lambda1Star;
    RtimesL[i][1] = R[i][1] * Lambda2Star;
    RtimesL[i][2] = R[i][2] * Lambda3Star;
  }

  for (PetscInt i = 0; i < 3; ++i) {
    for (PetscInt k = 0; k < 3; ++k) {
      PetscReal tmpL = 0.0, tmpR = 0.0;
      for (PetscInt j = 0; j < 3; ++j) {
        tmpL += RtimesL[i][j] * dV_dQl[j][k];
        tmpR += RtimesL[i][j] * dV_dQr[j][k];
      }
      Jl[i][k] += tmpL;
      Jr[i][k] += tmpR;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeSWERoeFluxJacobian(RiemannStateData *datal, RiemannStateData *datar, const PetscReal *sn, const PetscReal *cn, PetscReal *jup, PetscReal *jdn){

  PetscFunctionBeginUser;

  PetscReal *hl = datal->h;
  PetscReal *ul = datal->u;
  PetscReal *vl = datal->v;

  PetscReal *hr = datar->h;
  PetscReal *ur = datar->u;
  PetscReal *vr = datar->v;

  PetscAssert(datal->num_states == datar->num_states, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Size of data left and right of edges is not the same!");

  PetscInt num_states = datal->num_states;
  for (PetscInt i = 0; i < num_states; ++i) {

    PetscReal JperpL[3][3], JperpR[3][3], JdissL[3][3], JdissR[3][3];
    PetscCall(JacobianOfScalarFlux(hl[i], ul[i], vl[i], sn[i], cn[i], JperpL));
    PetscCall(JacobianOfScalarFlux(hr[i], ur[i], vr[i], sn[i], cn[i], JperpR));
    PetscCall(JacobianOfDissipiationTerm(hl[i], ul[i], vl[i], hr[i], ur[i], vr[i], sn[i], cn[i], JdissL, JdissR));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
#endif  // SWE_ROE_FLUX_PETSC_H
