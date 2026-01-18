#ifndef SWE_HLLC_PETSC_IMPL_H
#define SWE_HLLC_PETSC_IMPL_H

#include "swe_petsc_impl.h"  // ok to keep; helpers come from swe_hll_petsc_impl.h in the .c

// silence unused function warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"

// rotate velocity to (normal,tangential)
static inline void SWERotateToNT(const PetscReal u, const PetscReal v,
                                 const PetscReal sn, const PetscReal cn,
                                 PetscReal *un, PetscReal *ut) {
  *un = u * cn + v * sn;
  *ut = -u * sn + v * cn;
}

// rotate (un,ut) back to (u,v)
static inline void SWERotateToUV(const PetscReal un, const PetscReal ut,
                                 const PetscReal sn, const PetscReal cn,
                                 PetscReal *u, PetscReal *v) {
  *u = un * cn - ut * sn;
  *v = un * sn + ut * cn;
}

/// Computes flux based on an HLLC-type solver for 2D SWE
static PetscErrorCode ComputeSWEHLLCFlux(RiemannStateData *datal, RiemannStateData *datar,
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
    // Left/right physical fluxes
    PetscReal FL[3], FR[3];
    SWEPhysicalNormalFlux(hl[i], ul[i], vl[i], sn[i], cn[i], FL);
    SWEPhysicalNormalFlux(hr[i], ur[i], vr[i], sn[i], cn[i], FR);

    // Wave speed estimates
    PetscReal SL, SR;
    SWEHLLWaveSpeeds(hl[i], ul[i], vl[i], hr[i], ur[i], vr[i], sn[i], cn[i], &SL, &SR);
    amax[i] = PetscMax(PetscAbsReal(SL), PetscAbsReal(SR));

    // If all waves go one way, upwind
    if (SL >= 0.0) {
      fij[3 * i + 0] = FL[0];
      fij[3 * i + 1] = FL[1];
      fij[3 * i + 2] = FL[2];
      continue;
    }
    if (SR <= 0.0) {
      fij[3 * i + 0] = FR[0];
      fij[3 * i + 1] = FR[1];
      fij[3 * i + 2] = FR[2];
      continue;
    }

    // Rotate velocities to (un,ut)
    PetscReal unl, utl, unr, utr;
    SWERotateToNT(ul[i], vl[i], sn[i], cn[i], &unl, &utl);
    SWERotateToNT(ur[i], vr[i], sn[i], cn[i], &unr, &utr);

    // Pressure term p = 0.5*g*h^2
    const PetscReal pL = 0.5 * GRAVITY * hl[i] * hl[i];
    const PetscReal pR = 0.5 * GRAVITY * hr[i] * hr[i];

    // Middle wave speed S* in normal direction
    const PetscReal denom = hl[i] * (SL - unl) - hr[i] * (SR - unr);
    PetscReal Sstar;
    if (PetscAbsReal(denom) < 1e-14) {
      Sstar = 0.0;  // degenerate fallback
    } else {
      Sstar = (pR - pL + hl[i] * unl * (SL - unl) - hr[i] * unr * (SR - unr)) / denom;
    }

    // Star region depths
    const PetscReal hLstar = hl[i] * (SL - unl) / (SL - Sstar);
    const PetscReal hRstar = hr[i] * (SR - unr) / (SR - Sstar);

    // Tangential velocity constant across contact
    const PetscReal utLstar = utl;
    const PetscReal utRstar = utr;

    // Convert (S*, ut*) back to (u,v)
    PetscReal uLstar, vLstar, uRstar, vRstar;
    SWERotateToUV(Sstar, utLstar, sn[i], cn[i], &uLstar, &vLstar);
    SWERotateToUV(Sstar, utRstar, sn[i], cn[i], &uRstar, &vRstar);

    const PetscReal ULstar0 = hLstar;
    const PetscReal ULstar1 = hLstar * uLstar;
    const PetscReal ULstar2 = hLstar * vLstar;

    const PetscReal URstar0 = hRstar;
    const PetscReal URstar1 = hRstar * uRstar;
    const PetscReal URstar2 = hRstar * vRstar;

    const PetscReal UL0 = hl[i];
    const PetscReal UL1 = hl[i] * ul[i];
    const PetscReal UL2 = hl[i] * vl[i];

    const PetscReal UR0 = hr[i];
    const PetscReal UR1 = hr[i] * ur[i];
    const PetscReal UR2 = hr[i] * vr[i];

    if (Sstar >= 0.0) {
      fij[3 * i + 0] = FL[0] + SL * (ULstar0 - UL0);
      fij[3 * i + 1] = FL[1] + SL * (ULstar1 - UL1);
      fij[3 * i + 2] = FL[2] + SL * (ULstar2 - UL2);
    } else {
      fij[3 * i + 0] = FR[0] + SR * (URstar0 - UR0);
      fij[3 * i + 1] = FR[1] + SR * (URstar1 - UR1);
      fij[3 * i + 2] = FR[2] + SR * (URstar2 - UR2);
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic pop
#pragma clang diagnostic pop

#endif  // SWE_HLLC_PETSC_IMPL_H
