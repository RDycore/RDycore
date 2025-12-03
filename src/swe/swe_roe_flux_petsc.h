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

static PetscErrorCode ComputeLambdaAndDerivative (const PetscReal q1_l, const PetscReal q2_l, const PetscReal q3_l,
                                           const PetscReal q1_r, const PetscReal q2_r, const PetscReal q3_r,
                                           const PetscReal sn, const PetscReal cn,
                                           const PetscReal uhatperp, const PetscReal ahat,
                                           const PetscReal duhatperp_dql[3], const PetscReal dahat_dql[3],
                                           const PetscReal duhatperp_dqr[3], const PetscReal dahat_dqr[3],
                                           const PetscBool compute_Lambda1,
                                           PetscReal *lambdastar,
                                           PetscReal dlambdastar_dql[3],
                                           PetscReal dlambdastar_dqr[3]) {

  PetscFunctionBeginUser;

  PetscReal factor = (compute_Lambda1) ? -1 : 1;

  // compute lambda and its derivatives
  PetscReal uperp_l = q2_l / q1_l * cn + q3_l / q1_l * sn;
  PetscReal uperp_r = q2_r / q1_r * cn + q3_r / q1_r * sn;

  PetscReal lambda_l = uperp_l + factor * pow(GRAVITY * q1_l, 0.5);
  PetscReal lambda_r = uperp_r + factor * pow(GRAVITY * q1_r, 0.5);

  PetscReal lambda_l_dql[3] = { (-q2_l * cn - q3_l * sn) / (q1_l * q1_l) + factor * 0.5 * pow(GRAVITY / q1_l, 0.5),
                                 cn / q1_l,
                                 sn / q1_l};
  PetscReal lambda_r_dqr[3] = { (-q2_r * cn - q3_r * sn) / (q1_r * q1_r) + factor * 0.5 * pow(GRAVITY / q1_r, 0.5),
                                 cn / q1_r,
                                 sn / q1_r};

  // compute delta lambda and its derivatives
  PetscReal dellambda = 4.0 * (lambda_r - lambda_l);
  PetscReal ddellambda_dql[3] = { -4.0 * lambda_l_dql[0],
                                    -4.0 * lambda_l_dql[1],
                                    -4.0 * lambda_l_dql[2]};
  PetscReal ddellambda_dqr[3] = { 4.0 * lambda_r_dqr[0],
                                    4.0 * lambda_r_dqr[1],
                                    4.0 * lambda_r_dqr[2]};


  // compute lambdahat and its derivatives
  PetscReal lambdahat = (uhatperp - ahat);
  PetscReal dlambdahat_dql[3] = { duhatperp_dql[0] - dahat_dql[0],
                                   duhatperp_dql[1] - dahat_dql[1],
                                   duhatperp_dql[2] - dahat_dql[2]};
  PetscReal dlambdahat_dqr[3] = { duhatperp_dqr[0] - dahat_dqr[0],
                                   duhatperp_dqr[1] - dahat_dqr[1],
                                   duhatperp_dqr[2] - dahat_dqr[2]};

  if (fabs(lambdahat) < 0.5 * fabs(dellambda) ) {
    PetscReal factor = (lambdahat > 0.0) ? 1.0 : -1.0;

    // apply correction
    *lambdastar = factor * (lambdahat * lambdahat) / dellambda + dellambda / 4.0;

    dlambdastar_dql[0] = factor * ( (2.0 * lambdahat * dlambdahat_dql[0] * dellambda - lambdahat * lambdahat * ddellambda_dql[0]) / (dellambda * dellambda) + 0.25 * ddellambda_dql[0]);
    dlambdastar_dql[1] = factor * ( (2.0 * lambdahat * dlambdahat_dql[1] * dellambda - lambdahat * lambdahat * ddellambda_dql[1]) / (dellambda * dellambda) + 0.25 * ddellambda_dql[1]);
    dlambdastar_dql[2] = factor * ( (2.0 * lambdahat * dlambdahat_dql[2] * dellambda - lambdahat * lambdahat * ddellambda_dql[2]) / (dellambda * dellambda) + 0.25 * ddellambda_dql[2]);

    dlambdastar_dqr[0] = factor * ( (2.0 * lambdahat * dlambdahat_dqr[0] * dellambda - lambdahat * lambdahat * ddellambda_dqr[0]) / (dellambda * dellambda) + 0.25 * ddellambda_dqr[0]);
    dlambdastar_dqr[1] = factor * ( (2.0 * lambdahat * dlambdahat_dqr[1] * dellambda - lambdahat * lambdahat * ddellambda_dqr[1]) / (dellambda * dellambda) + 0.25 * ddellambda_dqr[1]);
    dlambdastar_dqr[2] = factor * ( (2.0 * lambdahat * dlambdahat_dqr[2] * dellambda - lambdahat * lambdahat * ddellambda_dqr[2]) / (dellambda * dellambda) + 0.25 * ddellambda_dqr[2]);

  } else {

    PetscReal factor = (lambdahat > 0.0) ? 1.0 : -1.0;

    // no correction
    *lambdastar = factor * lambdahat;

    dlambdastar_dql[0] = factor * dlambdahat_dql[0];
    dlambdastar_dql[1] = factor * dlambdahat_dql[1];
    dlambdastar_dql[2] = factor * dlambdahat_dql[2];

    dlambdastar_dqr[0] = factor * dlambdahat_dqr[0];
    dlambdastar_dqr[1] = factor * dlambdahat_dqr[1];
    dlambdastar_dqr[2] = factor * dlambdahat_dqr[2];
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode JacobianOfDissipiationTerm(const PetscReal hl, const PetscReal ul, const PetscReal vl, const PetscReal hr, const PetscReal ur,
                                                 const PetscReal vr, PetscReal sn, PetscReal cn, PetscReal Jl[3][3],PetscReal Jr[3][3]) {
  PetscFunctionBeginUser;

  PetscReal q1_l = hl;
  PetscReal q2_l = hl * ul;
  PetscReal q3_l = hl * vl;

  PetscReal q1_r = hr;
  PetscReal q2_r = hr * ur;
  PetscReal q3_r = hr * vr;

  PetscReal q1_l_sqrt = pow(q1_l, 0.5);
  PetscReal q1_r_sqrt = pow(q1_r, 0.5);

  // compute Roe averages: uhat and derivatives
  PetscReal uhat = ( q2_l / q1_l_sqrt + q2_r / q1_r_sqrt) / (q1_l_sqrt + q1_r_sqrt);
  PetscReal common_denom = 2.0 * q1_l_sqrt * q1_r_sqrt * pow(q1_l_sqrt + q1_r_sqrt, 2.0);
  PetscReal duhat_dql[3] = { - (q1_l * q2_r + 2.0 * q2_l * q1_l_sqrt * q1_r_sqrt + q2_l * q1_r) / (q1_l * common_denom),
                             1.0 / (q1_l_sqrt * (q1_l_sqrt + q1_r_sqrt)),
                             0.0};
  PetscReal duhat_dqr[3] = { - (q1_r * q2_l + 2.0 * q2_r * q1_l_sqrt * q1_r_sqrt + q2_r * q1_l) / (q1_r * common_denom),
                             1.0 / (q1_r_sqrt * (q1_l_sqrt + q1_r_sqrt)),
                             0.0};

    // compute Roe averages: vhat and its derivatives
  PetscReal vhat = ( q3_l / q1_l_sqrt + q3_r / q1_r_sqrt) / (q1_l_sqrt + q1_r_sqrt);
  PetscReal dvhat_dql[3] = { - (q1_l * q3_r + 2.0 * q3_l * q1_l_sqrt * q1_r_sqrt + q3_l * q1_r) / (q1_l * common_denom),
                             0.0,
                             1.0 / (q1_l_sqrt * (q1_l_sqrt + q1_r_sqrt))};
  PetscReal dvhat_dqr[3] = { - (q1_r * q3_l + 2.0 * q3_r * q1_l_sqrt * q1_r_sqrt + q3_r * q1_l) / (q1_r * common_denom),
                             0.0,
                             1.0 / (q1_r_sqrt * (q1_l_sqrt + q1_r_sqrt))};

  // compute ahat and its derivatives
  PetscReal ahat  = pow(0.5 * GRAVITY * (q1_l + q1_r), 0.5);
  PetscReal dahat_dql[3] = {0.25 * GRAVITY / ahat, 0.0, 0.0};
  PetscReal dahat_dqr[3] = {0.25 * GRAVITY / ahat, 0.0, 0.0};


  // compute duperp and its derivatives
  PetscReal duperp = (q2_r / q1_r - q2_l / q1_l) * cn + (q3_r / q1_r - q3_l / q1_l) * sn;
  PetscReal dduperp_dql[3] = { (q2_l * cn + q3_l * sn) / (q1_l * q1_l),
                               - cn / q1_l,
                               - sn / q1_l};
  PetscReal dduperp_dqr[3] = { -(q2_r * cn + q3_r * sn) / (q1_r * q1_r),
                               cn / q1_r,
                               sn / q1_r};

  // compute dupar and its derivatives
  PetscReal dupar = -(q2_r / q1_r - q2_l / q1_l) * sn + (q3_r / q1_r - q3_l / q1_l) * cn;
  PetscReal ddupar_dql[3] = { -(q2_l * sn + q3_l * cn) / (q1_l * q1_l),
                               sn / q1_l,
                               cn / q1_l};
  PetscReal ddupar_dqr[3] = { (q2_r * sn + q3_r * cn) / (q1_r * q1_r),
                               -sn / q1_r,
                               -cn / q1_r};

  // compute uhatperp and its derivatives
  PetscReal uhatperp = uhat * cn + vhat * sn;
  PetscReal duhatperp_dql[3] = { duhat_dql[0] * cn + dvhat_dql[0] * sn,
                                 duhat_dql[1] * cn + dvhat_dql[1] * sn,
                                 duhat_dql[2] * cn + dvhat_dql[2] * sn};
  PetscReal duhatperp_dqr[3] = { duhat_dqr[0] * cn + dvhat_dqr[0] * sn,
                                 duhat_dqr[1] * cn + dvhat_dqr[1] * sn,
                                 duhat_dqr[2] * cn + dvhat_dqr[2] * sn};

  // compute dV and its derivatives
  PetscReal dV[3], dV_dql[3][3], dV_dqr[3][3];
  {
    // compute Roe averages: hhat and its derivatives
    PetscReal hhat  = pow(q1_l * q1_r, 0.5);
    PetscReal dhat_dql[3] = {pow(q1_r / q1_l, 0.5) * 0.5, 0.0, 0.0};
    PetscReal dhat_dqr[3] = {pow(q1_l / q1_r, 0.5) * 0.5, 0.0, 0.0};

    PetscReal dh = q1_r - q1_l;
    PetscReal dh_dql[3] = { -1.0, 0.0, 0.0};
    PetscReal dh_dqr[3] = { 1.0, 0.0, 0.0};

    dV[0] = 0.5 * (dh - hhat * duperp / ahat);
    dV[1] = hhat * dupar;
    dV[2] = 0.5 * (dh + hhat * duperp / ahat);

    for (PetscInt k = 0; k < 3; ++k) {
      dV_dql[0][k] = 0.5 * (dh_dql[k] - (dhat_dql[k] * duperp + hhat * dduperp_dql[k]) / ahat + hhat * duperp * dahat_dql[k] / (ahat * ahat));
      dV_dql[1][k] = dhat_dql[k] * dupar + hhat * ddupar_dql[k];
      dV_dql[2][k] = 0.5 * (dh_dql[k] + (dhat_dql[k] * duperp + hhat * dduperp_dql[k]) / ahat - hhat * duperp * dahat_dql[k] / (ahat * ahat));

      dV_dqr[0][k] = 0.5 * (dh_dqr[k] - (dhat_dqr[k] * duperp + hhat * dduperp_dqr[k]) / ahat + hhat * duperp * dahat_dqr[k] / (ahat * ahat));
      dV_dqr[1][k] = dhat_dqr[k] * dupar + hhat * ddupar_dqr[k];
      dV_dqr[2][k] = 0.5 * (dh_dqr[k] + (dhat_dqr[k] * duperp + hhat * dduperp_dqr[k]) / ahat - hhat * duperp * dahat_dqr[k] / (ahat * ahat));
    }
  }

  // compute lambda_star and its derivatives
  PetscBool compute_Lambda1;

  PetscReal lambda1star, dlambda1star_dql[3], dlambda1star_dqr[3];
  compute_Lambda1 = PETSC_FALSE;
  PetscCall(ComputeLambdaAndDerivative (q1_l, q2_l, q3_l,
                                q1_r, q2_r, q3_r,
                                sn, cn,
                                uhatperp, ahat,
                                duhatperp_dql, dahat_dql,
                                duhatperp_dqr, dahat_dqr,
                                compute_Lambda1,
                                &lambda1star,
                                dlambda1star_dql, dlambda1star_dqr));

  PetscReal lambda3star, dlambda3star_dql[3], dlambda3star_dqr[3];
  compute_Lambda1 = PETSC_FALSE;
  PetscCall(ComputeLambdaAndDerivative (q1_l, q2_l, q3_l,
                                q1_r, q2_r, q3_r,
                                sn, cn,
                                uhatperp, ahat,
                                duhatperp_dql, dahat_dql,
                                duhatperp_dqr, dahat_dqr,
                                compute_Lambda1,
                                &lambda3star,
                                dlambda3star_dql, dlambda3star_dqr));

  PetscReal lambda2star, dlambda2star_dql[3], dlambda2star_dqr[3];
  PetscReal factor = (uhatperp > 0.0) ? 1.0 : -1.0;
  lambda2star = factor * uhatperp;
  dlambda2star_dql[0] = factor * duhatperp_dql[0];
  dlambda2star_dql[1] = factor * duhatperp_dql[1];
  dlambda2star_dql[2] = factor * duhatperp_dql[2];

  dlambda2star_dqr[0] = factor * duhatperp_dqr[0];
  dlambda2star_dqr[1] = factor * duhatperp_dqr[1];
  dlambda2star_dqr[2] = factor * duhatperp_dqr[2];

  PetscReal R[3][3] = {
    {1.0,          0.0,           1.0},
    {uhat - ahat * cn, -sn,        uhat + ahat * cn},
    {vhat - ahat * sn,  cn,        vhat + ahat * sn}
  };

  PetscReal G_l[3][3][3], G_r[3][3][3], H_l[3][3][3], H_r[3][3][3];

  // initialize derivatives of R to zero
  for (PetscInt i = 0; i < 3; ++i) {
    for (PetscInt j = 0; j < 3; ++j) {
      for (PetscInt k = 0; k < 3; ++k) {
        G_l[i][j][k] = 0.0;
        G_r[i][j][k] = 0.0;
        H_l[i][j][k] = 0.0;
        H_r[i][j][k] = 0.0;
      }
    }
  }

  // only fill diagonals since others will be multiplied by zero later
  for (PetscInt k = 0; k < 3; k++) {
    PetscReal dR_dql = dvhat_dql[k] + dahat_dql[k] * sn;
    PetscReal dR_dqr = dvhat_dqr[k] + dahat_dqr[k] * sn;

    G_l[2][2][k] = dR_dql * lambda3star;
    G_r[2][2][k] = dR_dqr * lambda3star;

    H_l[0][0][k] = R[0][0] * dlambda1star_dql[k];
    H_l[1][1][k] = R[1][1] * dlambda2star_dql[k];
    H_l[2][2][k] = R[2][2] * dlambda3star_dql[k];
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
      PetscReal tmp_l = 0.0, tmp_r = 0.0;
      for (PetscInt j = 0; j < 3; ++j) {
        tmp_l += G_l[i][j][k] * dV[k];
        tmp_r += G_r[i][j][k] * dV[k];

        tmp_l += H_l[i][j][k] * dV[k];
        tmp_r += H_r[i][j][k] * dV[k];
      }
      Jl[i][k] += tmp_l;
      Jr[i][k] += tmp_r;
    }
  }

  PetscReal RL[3][3];
  for (PetscInt i = 0; i < 3; ++i) {
    RL[i][0] = R[i][0] * lambda1star;
    RL[i][1] = R[i][1] * lambda2star;
    RL[i][2] = R[i][2] * lambda3star;
  }

  for (PetscInt i = 0; i < 3; ++i) {
    for (PetscInt k = 0; k < 3; ++k) {
      PetscReal tmp_l = 0.0, tmp_r = 0.0;
      for (PetscInt j = 0; j < 3; ++j) {
        tmp_l += RL[i][j] * dV_dql[j][k];
        tmp_r += RL[i][j] * dV_dqr[j][k];
      }
      Jl[i][k] += tmp_l;
      Jr[i][k] += tmp_r;
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

    PetscReal Jperp_l[3][3], Jperp_r[3][3], Jdiss_l[3][3], Jdiss_r[3][3];
    PetscCall(JacobianOfScalarFlux(hl[i], ul[i], vl[i], sn[i], cn[i], Jperp_l));
    PetscCall(JacobianOfScalarFlux(hr[i], ur[i], vr[i], sn[i], cn[i], Jperp_r));
    PetscCall(JacobianOfDissipiationTerm(hl[i], ul[i], vl[i], hr[i], ur[i], vr[i], sn[i], cn[i], Jdiss_l, Jdiss_r));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
#endif  // SWE_ROE_FLUX_PETSC_H
