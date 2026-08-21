#ifndef SWE_ROE_FLUX_PETSC_H
#define SWE_ROE_FLUX_PETSC_H

#include <private/rdymathimpl.h>  // Square

#include "swe_types_petsc.h"

// Qualifier for the per-edge flux math below, which is shared VERBATIM by the
// host loops and the Kokkos device RHS kernels: plain C sees static inline;
// the .kokkos.cxx TU defines RDY_MATH_FN to `static KOKKOS_INLINE_FUNCTION`
// before including this header. The functions are pure scalar math (no PETSc
// objects, no error paths), so they carry no PetscErrorCode plumbing.
#ifndef RDY_MATH_FN
#define RDY_MATH_FN static inline
#endif

// silence unused function warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"

// Computes eigenvalues lambda, right eigenvectors R, parameter change dW, and
// the maximum wave speed for the shallow water equations
RDY_MATH_FN void ComputeSWERoeEigenspectrum(const PetscReal hl, const PetscReal ul, const PetscReal vl, const PetscReal hr, const PetscReal ur,
                                            const PetscReal vr, PetscReal sn, PetscReal cn, PetscReal lambda[3], PetscReal R[3][3],
                                            PetscReal dW[3], PetscReal *amax) {
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
}

// Computes the Roe flux fij[3] and maximum wave speed for ONE edge from the
// left/right primitive states -- the per-edge body of ComputeSWERoeFlux(),
// shared verbatim with the Kokkos device RHS kernels.
RDY_MATH_FN void ComputeSWERoeFluxEdge(const PetscReal hl, const PetscReal ul, const PetscReal vl, const PetscReal hr, const PetscReal ur,
                                       const PetscReal vr, const PetscReal sn, const PetscReal cn, PetscReal fij[3], PetscReal *amax) {
  // compute eigenspectrum
  PetscReal A[3], R[3][3], dW[3];
  ComputeSWERoeEigenspectrum(hl, ul, vl, hr, ur, vr, sn, cn, A, R, dW, amax);

  // compute interface fluxes
  PetscReal uperpl = ul * cn + vl * sn;
  PetscReal uperpr = ur * cn + vr * sn;
  PetscReal FL[3]  = {
       uperpl * hl,
       ul * uperpl * hl + 0.5 * GRAVITY * hl * hl * cn,
       vl * uperpl * hl + 0.5 * GRAVITY * hl * hl * sn,
  };
  PetscReal FR[3] = {
      uperpr * hr,
      ur * uperpr * hr + 0.5 * GRAVITY * hr * hr * cn,
      vr * uperpr * hr + 0.5 * GRAVITY * hr * hr * sn,
  };

  // fij = 0.5*(FL + FR - matmul(R,matmul(A,dW))
  fij[0] = 0.5 * (FL[0] + FR[0] - R[0][0] * A[0] * dW[0] - R[0][1] * A[1] * dW[1] - R[0][2] * A[2] * dW[2]);
  fij[1] = 0.5 * (FL[1] + FR[1] - R[1][0] * A[0] * dW[0] - R[1][1] * A[1] * dW[1] - R[1][2] * A[2] * dW[2]);
  fij[2] = 0.5 * (FL[2] + FR[2] - R[2][0] * A[0] * dW[0] - R[2][1] * A[1] * dW[1] - R[2][2] * A[2] * dW[2]);
}

// Computes primitive velocities (u, v) from one conservative state with the
// ANUGA regularization and tiny_h dry cutoff -- the per-state body of
// ComputeRiemannVelocities() in swe_petsc.c, shared with the device kernels.
RDY_MATH_FN void ComputeSWERiemannVelocity(const PetscReal h, const PetscReal hu, const PetscReal hv, const PetscReal tiny_h,
                                           const PetscReal h_anuga, PetscReal *u, PetscReal *v) {
  if (h < tiny_h) {
    *u = 0.0;
    *v = 0.0;
  } else {
    PetscReal denom = Square(h) + Square(h_anuga);
    *u              = hu * h / denom;
    *v              = hv * h / denom;
  }
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
    ComputeSWERoeFluxEdge(hl[i], ul[i], vl[i], hr[i], ur[i], vr[i], sn[i], cn[i], &fij[3 * i], &amax[i]);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
#endif  // SWE_ROE_FLUX_PETSC_H
