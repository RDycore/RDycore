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

    PetscReal Jperp_l[3][3], Jperp_r[3][3];
    PetscCall(JacobianOfScalarFlux(hl[i], ul[i], vl[i], sn[i], cn[i], Jperp_l));
    PetscCall(JacobianOfScalarFlux(hr[i], ur[i], vr[i], sn[i], cn[i], Jperp_r));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
#endif  // SWE_ROE_FLUX_PETSC_H
