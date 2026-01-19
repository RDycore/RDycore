#ifndef SWE_HLL_PETSC_IMPL_H
#define SWE_HLL_PETSC_IMPL_H

#include "swe_petsc_impl.h"

// silence unused function warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"

// physical flux in direction n = (cn, sn) for SWE state (h,u,v)
static inline void SWEPhysicalNormalFlux(const PetscReal h, const PetscReal u, const PetscReal v,
                                         const PetscReal sn, const PetscReal cn, PetscReal F[3]) {
  const PetscReal un = u * cn + v * sn;

  F[0] = h * un;
  F[1] = h * u * un + 0.5 * GRAVITY * h * h * cn;
  F[2] = h * v * un + 0.5 * GRAVITY * h * h * sn;
}

// estimate HLL wave speeds SL, SR
// Uses common shallow-water estimates based on normal velocity and c = sqrt(g h).
static inline void SWEHLLWaveSpeeds(const PetscReal hl, const PetscReal ul, const PetscReal vl,
                                    const PetscReal hr, const PetscReal ur, const PetscReal vr,
                                    const PetscReal sn, const PetscReal cn,
                                    PetscReal *SL, PetscReal *SR) {
  const PetscReal unl = ul * cn + vl * sn;
  const PetscReal unr = ur * cn + vr * sn;

  const PetscReal cl = PetscSqrtReal(GRAVITY * hl);
  const PetscReal cr = PetscSqrtReal(GRAVITY * hr);

  *SL = PetscMin(unl - cl, unr - cr);
  *SR = PetscMax(unl + cl, unr + cr);
}

/// Computes flux based on HLL solver (2-wave approximate Riemann solver)
static PetscErrorCode ComputeSWEHLLFlux(RiemannStateData *datal, RiemannStateData *datar,
                                       const PetscReal *sn, const PetscReal *cn,
                                       PetscReal *fij, PetscReal *amax) {
  PetscFunctionBeginUser;

  PetscReal *hl = datal->h;
  PetscReal *ul = datal->u;
  PetscReal *vl = datal->v;

  PetscReal *hr = datar->h;
  PetscReal *ur = datar->u;
  PetscReal *vr = datar->v;

  PetscAssert(datal->num_states == datar->num_states, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ,
              "Size of data left and right of edges is not the same!");

  const PetscInt num_states = datal->num_states;

  for (PetscInt i = 0; i < num_states; ++i) {
    PetscReal FL[3], FR[3];
    SWEPhysicalNormalFlux(hl[i], ul[i], vl[i], sn[i], cn[i], FL);
    SWEPhysicalNormalFlux(hr[i], ur[i], vr[i], sn[i], cn[i], FR);

    PetscReal SL, SR;
    SWEHLLWaveSpeeds(hl[i], ul[i], vl[i], hr[i], ur[i], vr[i], sn[i], cn[i], &SL, &SR);

    amax[i] = PetscMax(PetscAbsReal(SL), PetscAbsReal(SR));

    if (SL >= 0.0) {
      fij[3 * i + 0] = FL[0];
      fij[3 * i + 1] = FL[1];
      fij[3 * i + 2] = FL[2];
    } else if (SR <= 0.0) {
      fij[3 * i + 0] = FR[0];
      fij[3 * i + 1] = FR[1];
      fij[3 * i + 2] = FR[2];
    } else {
      const PetscReal UL0 = hl[i];
      const PetscReal UL1 = hl[i] * ul[i];
      const PetscReal UL2 = hl[i] * vl[i];

      const PetscReal UR0 = hr[i];
      const PetscReal UR1 = hr[i] * ur[i];
      const PetscReal UR2 = hr[i] * vr[i];

      const PetscReal inv = 1.0 / (SR - SL);

      fij[3 * i + 0] = (SR * FL[0] - SL * FR[0] + SL * SR * (UR0 - UL0)) * inv;
      fij[3 * i + 1] = (SR * FL[1] - SL * FR[1] + SL * SR * (UR1 - UL1)) * inv;
      fij[3 * i + 2] = (SR * FL[2] - SL * FR[2] + SL * SR * (UR2 - UL2)) * inv;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic pop
#pragma clang diagnostic pop

#endif  // SWE_HLL_PETSC_IMPL_H
