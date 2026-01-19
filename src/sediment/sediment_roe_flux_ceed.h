#ifndef SEDIMENT_ROE_FLUX_CEED_H
#define SEDIMENT_ROE_FLUX_CEED_H

#include "../swe/swe_fluxes_ceed.h"

// we disable compiler warnings for implicitly-declared math functions known to
// the JIT compiler
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wimplicit-function-declaration"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-function-declaration"

/// computes the flux across an edge using Roe's approximate Riemann solver
/// for flow and sediment transport
CEED_QFUNCTION_HELPER void SedimentRiemannFlux_Roe(const CeedScalar gravity, const CeedScalar tiny_h, SedimentState qL, SedimentState qR,
                                                   CeedScalar sn, CeedScalar cn, CeedInt flow_ndof, CeedInt sed_ndof, CeedScalar flux[],
                                                   CeedScalar *amax) {
  const CeedScalar hl = qL.h, hr = qR.h;
  const CeedScalar ul = SafeDiv(qL.hu, hl, hl, tiny_h);
  const CeedScalar vl = SafeDiv(qL.hv, hl, hl, tiny_h);
  CeedScalar       cil[MAX_NUM_SEDIMENT_CLASSES], cir[MAX_NUM_SEDIMENT_CLASSES];
  for (CeedInt j = 0; j < sed_ndof; ++j) {
    cil[j] = SafeDiv(qL.hci[j], hl, hl, tiny_h);
    cir[j] = SafeDiv(qR.hci[j], hr, hl, tiny_h);
  }
  const CeedScalar ur = SafeDiv(qR.hu, hr, hr, tiny_h);
  const CeedScalar vr = SafeDiv(qR.hv, hr, hr, tiny_h);

  // compute the eigenspectrum for the shallow water equations
  CeedScalar A_swe[3], R_swe[3][3], dW_swe[3], amax_swe;
  ComputeSWEEigenspectrum_Roe(hl, ul, vl, hr, ur, vr, sn, cn, gravity, A_swe, R_swe, dW_swe, &amax_swe);

  // compute sediment contributions
  CeedScalar cihat[MAX_NUM_FIELD_COMPONENTS]                       = {0};
  CeedScalar dch[MAX_NUM_FIELD_COMPONENTS]                         = {0};
  CeedScalar dW[MAX_NUM_FIELD_COMPONENTS]                          = {0};
  CeedScalar R[MAX_NUM_FIELD_COMPONENTS][MAX_NUM_FIELD_COMPONENTS] = {0};
  CeedScalar A[MAX_NUM_FIELD_COMPONENTS]                           = {0};
  CeedScalar FL[MAX_NUM_FIELD_COMPONENTS]                          = {0};
  CeedScalar FR[MAX_NUM_FIELD_COMPONENTS]                          = {0};

  CeedScalar duml   = sqrt(hl);
  CeedScalar dumr   = sqrt(hr);
  CeedScalar dh     = hr - hl;
  CeedScalar uperpl = ul * cn + vl * sn;
  CeedScalar uperpr = ur * cn + vr * sn;

  for (CeedInt j = 0; j < sed_ndof; j++) {
    cihat[j] = (duml * cil[j] + dumr * cir[j]) / (duml + dumr);
    dch[j]   = cir[j] * hr - cil[j] * hl;
  }

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      R[i][j] = R_swe[i][j];
    }
  }
  for (CeedInt j = 0; j < sed_ndof; j++) {
    R[j + 3][0]     = cihat[j];
    R[j + 3][2]     = cihat[j];
    R[j + 3][j + 3] = 1.0;
  }

  A[0] = A_swe[0];
  A[1] = A_swe[1];
  A[2] = A_swe[2];
  for (CeedInt j = 0; j < sed_ndof; j++) {
    A[j + 3] = A[1];
  }

  dW[0] = dW_swe[0];
  dW[1] = dW_swe[1];
  dW[2] = dW_swe[2];
  for (CeedInt j = 0; j < sed_ndof; j++) {
    dW[j + 3] = dch[j] - cihat[j] * dh;
  }

  // compute interface fluxes
  FL[0] = uperpl * hl;
  FL[1] = ul * uperpl * hl + 0.5 * gravity * hl * hl * cn;
  FL[2] = vl * uperpl * hl + 0.5 * gravity * hl * hl * sn;

  FR[0] = uperpr * hr;
  FR[1] = ur * uperpr * hr + 0.5 * gravity * hr * hr * cn;
  FR[2] = vr * uperpr * hr + 0.5 * gravity * hr * hr * sn;

  for (CeedInt j = 0; j < sed_ndof; j++) {
    FL[j + 3] = hl * uperpl * cil[j];
    FR[j + 3] = hr * uperpr * cir[j];
  }

  // flux = 0.5*(FL + FR - matmul(R,matmul(A,dW))
  CeedInt soln_ncomp = flow_ndof + sed_ndof;
  for (CeedInt dof1 = 0; dof1 < soln_ncomp; dof1++) {
    flux[dof1] = 0.5 * (FL[dof1] + FR[dof1]);

    for (CeedInt dof2 = 0; dof2 < soln_ncomp; dof2++) {
      flux[dof1] = flux[dof1] - 0.5 * R[dof1][dof2] * A[dof2] * dW[dof2];
    }
  }

  // max wave speed
  *amax = amax_swe;
}
// ---- HLL and HLLC sediment Riemann solvers (projected 2-D SWE) ----
// We solve in the face-normal frame for [h, m_n] with pressure p = 0.5*g*h^2.
// Tangential momentum m_t and each sediment tracer h*c_i are treated as
// passively-advected scalars across acoustic waves.

static inline void SedimentRiemannFlux_HLL(const CeedScalar gravity,
                                           const CeedScalar tiny_h,
                                           SedimentState qL, SedimentState qR,
                                           CeedScalar sn, CeedScalar cn,
                                           CeedInt flow_ndof, CeedInt sed_ndof,
                                           CeedScalar flux[], CeedScalar *amax)
{
  // Left / Right states
  const CeedScalar hL = qL.h, hR = qR.h;
  const CeedScalar uL = SafeDiv(qL.hu, hL, hL, tiny_h);
  const CeedScalar vL = SafeDiv(qL.hv, hL, hL, tiny_h);
  const CeedScalar uR = SafeDiv(qR.hu, hR, hR, tiny_h);
  const CeedScalar vR = SafeDiv(qR.hv, hR, hR, tiny_h);

  // Normal/tangential velocities and momenta at the face
  const CeedScalar unL = uL*cn + vL*sn;                     // normal velocity
  const CeedScalar unR = uR*cn + vR*sn;
  const CeedScalar mtL = -qL.hu*sn + qL.hv*cn;              // tangential momentum (conserved)
  const CeedScalar mtR = -qR.hu*sn + qR.hv*cn;

  const CeedScalar cL  = sqrt(gravity * fmax(hL, 0.0));     // wave speeds
  const CeedScalar cR  = sqrt(gravity * fmax(hR, 0.0));
  const CeedScalar SL  = fmin(unL - cL, unR - cR);
  const CeedScalar SR  = fmax(unL + cL, unR + cR);

  // Conservative variables in (h, m_n, m_t, h c_i) representation
  const CeedScalar mnL = qL.hu*cn + qL.hv*sn;               // normal momentum = h*un
  const CeedScalar mnR = qR.hu*cn + qR.hv*sn;

  // Pressures
  const CeedScalar pL = 0.5 * gravity * hL * hL;
  const CeedScalar pR = 0.5 * gravity * hR * hR;

  // Physical fluxes in the normal direction for this representation
  // F_h = un*h, F_mn = un*mn + p, F_mt = un*mt, F_hci = un*(hci)
  CeedScalar FL_h   = unL * hL;
  CeedScalar FL_mn  = unL * mnL + pL;
  CeedScalar FL_mt  = unL * mtL;

  CeedScalar FR_h   = unR * hR;
  CeedScalar FR_mn  = unR * mnR + pR;
  CeedScalar FR_mt  = unR * mtR;

  // Degenerate cases: dry states or no intermediate fan
  if (SL >= 0) {
    // Pure left flux
    flux[0] = FL_h;
    // Map back to (hu,hv): F(hu) = F_mn*cn - F_mt*sn, F(hv) = F_mn*sn + F_mt*cn
    flux[1] = FL_mn*cn - FL_mt*sn;
    flux[2] = FL_mn*sn + FL_mt*cn;
    for (CeedInt j = 0; j < sed_ndof; ++j) {
      const CeedScalar hciL = qL.hci[j];
      flux[3 + j] = unL * hciL;
    }
    *amax = fabs(SL);
    return;
  } else if (SR <= 0) {
    // Pure right flux
    flux[0] = FR_h;
    flux[1] = FR_mn*cn - FR_mt*sn;
    flux[2] = FR_mn*sn + FR_mt*cn;
    for (CeedInt j = 0; j < sed_ndof; ++j) {
      const CeedScalar hciR = qR.hci[j];
      flux[3 + j] = unR * hciR;
    }
    *amax = fabs(SR);
    return;
  }

  // HLL middle state
  const CeedScalar denom = (SR - SL);
  const CeedScalar UL_h  = hL,    UR_h  = hR;
  const CeedScalar UL_mn = mnL,   UR_mn = mnR;
  const CeedScalar UL_mt = mtL,   UR_mt = mtR;

  const CeedScalar FHLL_h  = (SR*FL_h  - SL*FR_h  + SL*SR*(UR_h  - UL_h ))/denom;
  const CeedScalar FHLL_mn = (SR*FL_mn - SL*FR_mn + SL*SR*(UR_mn - UL_mn))/denom;
  const CeedScalar FHLL_mt = (SR*FL_mt - SL*FR_mt + SL*SR*(UR_mt - UL_mt))/denom;

  flux[0] = FHLL_h;
  flux[1] = FHLL_mn*cn - FHLL_mt*sn;
  flux[2] = FHLL_mn*sn + FHLL_mt*cn;

  for (CeedInt j = 0; j < sed_ndof; ++j) {
    const CeedScalar UL_hci = qL.hci[j];
    const CeedScalar UR_hci = qR.hci[j];
    const CeedScalar FL_hci = unL * UL_hci;
    const CeedScalar FR_hci = unR * UR_hci;
    const CeedScalar FHLL_hci = (SR*FL_hci - SL*FR_hci + SL*SR*(UR_hci - UL_hci))/denom;
    flux[3 + j] = FHLL_hci;
  }

  *amax = fmax(fabs(SL), fabs(SR));
}

static inline void SedimentRiemannFlux_HLLC(const CeedScalar gravity,
                                            const CeedScalar tiny_h,
                                            SedimentState qL, SedimentState qR,
                                            CeedScalar sn, CeedScalar cn,
                                            CeedInt flow_ndof, CeedInt sed_ndof,
                                            CeedScalar flux[], CeedScalar *amax)
{
  // Left / Right states
  const CeedScalar hL = qL.h, hR = qR.h;
  const CeedScalar uL = SafeDiv(qL.hu, hL, hL, tiny_h);
  const CeedScalar vL = SafeDiv(qL.hv, hL, hL, tiny_h);
  const CeedScalar uR = SafeDiv(qR.hu, hR, hR, tiny_h);
  const CeedScalar vR = SafeDiv(qR.hv, hR, hR, tiny_h);

  // Normal/tangential velocities and momenta
  const CeedScalar unL = uL*cn + vL*sn;
  const CeedScalar unR = uR*cn + vR*sn;

  const CeedScalar mnL = qL.hu*cn + qL.hv*sn;              // normal momentum
  const CeedScalar mnR = qR.hu*cn + qR.hv*sn;
  const CeedScalar mtL = -qL.hu*sn + qL.hv*cn;             // tangential momentum (passive)
  const CeedScalar mtR = -qR.hu*sn + qR.hv*cn;

  const CeedScalar cL  = sqrt(gravity * fmax(hL, 0.0));
  const CeedScalar cR  = sqrt(gravity * fmax(hR, 0.0));
  const CeedScalar SL  = fmin(unL - cL, unR - cR);
  const CeedScalar SR  = fmax(unL + cL, unR + cR);

  // Pressures
  const CeedScalar pL = 0.5 * gravity * hL * hL;
  const CeedScalar pR = 0.5 * gravity * hR * hR;

  // Physical fluxes (normal frame)
  const CeedScalar FL_h  = unL * hL;
  const CeedScalar FL_mn = unL * mnL + pL;
  const CeedScalar FL_mt = unL * mtL;

  const CeedScalar FR_h  = unR * hR;
  const CeedScalar FR_mn = unR * mnR + pR;
  const CeedScalar FR_mt = unR * mtR;

  // Upwind regions that bypass star construction
  if (SL >= 0) {
    flux[0] = FL_h;
    flux[1] = FL_mn*cn - FL_mt*sn;
    flux[2] = FL_mn*sn + FL_mt*cn;
    for (CeedInt j = 0; j < sed_ndof; ++j) {
      flux[3 + j] = unL * qL.hci[j];
    }
    *amax = fabs(SL);
    return;
  }
  if (SR <= 0) {
    flux[0] = FR_h;
    flux[1] = FR_mn*cn - FR_mt*sn;
    flux[2] = FR_mn*sn + FR_mt*cn;
    for (CeedInt j = 0; j < sed_ndof; ++j) {
      flux[3 + j] = unR * qR.hci[j];
    }
    *amax = fabs(SR);
    return;
  }

  // Contact speed S* (Toro-style, adapted to shallow water)
  const CeedScalar numS = (pR - pL)
                        + hL * unL * (SL - unL)
                        - hR * unR * (SR - unR);
  const CeedScalar denS = hL * (SL - unL) - hR * (SR - unR);
  const CeedScalar Sstar = numS / denS;

  // Star states (left and right)
  // h* = h (S - u_n) / (S - S*)
  const CeedScalar hLstar = hL * (SL - unL) / (SL - Sstar);
  const CeedScalar hRstar = hR * (SR - unR) / (SR - Sstar);

  // mn* = h* S*
  const CeedScalar mnLstar = hLstar * Sstar;
  const CeedScalar mnRstar = hRstar * Sstar;

  // mt behaves like a passive scalar with the mass wave
  const CeedScalar mtLstar = mtL * (SL - unL) / (SL - Sstar);
  const CeedScalar mtRstar = mtR * (SR - unR) / (SR - Sstar);

  // Each sediment conservative variable behaves like a passive scalar:
  // h c_i * = (h c_i) * (S - u_n)/(S - S*)
  // (equivalently, c_i* = c_i and h* as above)
  // Build flux depending on where 0 lies relative to SL, S*, SR.
  if (Sstar >= 0) {
    // Use left star flux: F*_L = F_L + S_L (U*_L - U_L)
    const CeedScalar F_h   = FL_h  + SL * (hLstar  - hL);
    const CeedScalar F_mn  = FL_mn + SL * (mnLstar - mnL);
    const CeedScalar F_mt  = FL_mt + SL * (mtLstar - mtL);

    flux[0] = F_h;
    flux[1] = F_mn*cn - F_mt*sn;
    flux[2] = F_mn*sn + F_mt*cn;

    for (CeedInt j = 0; j < sed_ndof; ++j) {
      const CeedScalar hciL = qL.hci[j];
      const CeedScalar hciLstar = hciL * (SL - unL) / (SL - Sstar);
      const CeedScalar F_hci = (unL * hciL) + SL * (hciLstar - hciL);
      flux[3 + j] = F_hci;
    }
  } else {
    // Use right star flux: F*_R = F_R + S_R (U*_R - U_R)
    const CeedScalar F_h   = FR_h  + SR * (hRstar  - hR);
    const CeedScalar F_mn  = FR_mn + SR * (mnRstar - mnR);
    const CeedScalar F_mt  = FR_mt + SR * (mtRstar - mtR);

    flux[0] = F_h;
    flux[1] = F_mn*cn - F_mt*sn;
    flux[2] = F_mn*sn + F_mt*cn;

    for (CeedInt j = 0; j < sed_ndof; ++j) {
      const CeedScalar hciR = qR.hci[j];
      const CeedScalar hciRstar = hciR * (SR - unR) / (SR - Sstar);
      const CeedScalar F_hci = (unR * hciR) + SR * (hciRstar - hciR);
      flux[3 + j] = F_hci;
    }
  }

  *amax = fmax(fabs(SL), fabs(SR));
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop

#endif
