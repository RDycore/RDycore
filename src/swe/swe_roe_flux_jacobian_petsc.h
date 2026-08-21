#ifndef SWE_ROE_FLUX_JACOBIAN_PETSC_H
#define SWE_ROE_FLUX_JACOBIAN_PETSC_H

// Analytic Jacobian blocks of the Roe interface flux with respect to the
// conservative states (h, hu, hv) of the two adjacent cells.
//
// The flux implemented in swe_roe_flux_petsc.h is
//   F(uL, uR) = 1/2 (F_phys(qL) + F_phys(qR) - R |Lambda| dW),
// where q = (h, u, v) are primitives reconstructed from conservative u with
// the ANUGA regularization (u = hu h / (h^2 + h_anuga^2), zero when
// h < tiny_h), R/|Lambda| are the Roe eigenvector/eigenvalue matrices at the
// Roe-averaged state, and dW are wave strengths, linear in the primitive
// differences (dh, du, dv).
//
// The Jacobian blocks are EXACT (increment 2c): they are assembled from a
// hand-written forward-mode differential of the flux computation that mirrors
// ComputeSWERoeEigenspectrum() and the flux assembly line by line -- Roe
// averages, eigenvectors, eigenvalues including both branches of the
// critical-flow (entropy) fix, wave strengths, and the physical fluxes --
// followed by the primitive-reconstruction chain rule. At the |.| and
// branch-switch points the consistent one-sided derivative is taken
// (subgradient), matching the code's own branch selection.

#include <private/rdymathimpl.h>

#include "swe_roe_flux_petsc.h"  // ComputeSWERoeEigenspectrum, GRAVITY (via swe_types_petsc.h)

// Qualifier for the flux-Jacobian math below, which is shared VERBATIM by the
// host assembly loop and the Kokkos device-assembly kernels: plain C sees
// static inline; the .kokkos.cxx TU defines RDY_MATH_FN to
// `static KOKKOS_INLINE_FUNCTION` before including this header. The functions
// are pure scalar math (no PETSc objects, no error paths), so they carry no
// PetscErrorCode plumbing.
#ifndef RDY_MATH_FN
#define RDY_MATH_FN static inline
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"

// Computes primitives q = (h, u, v) and the reconstruction Jacobian
// P = dq/d(h, hu, hv) for one cell state, mirroring ComputeRiemannVelocities()
// in swe_petsc.c (ANUGA regularization with the tiny_h dry cutoff).
RDY_MATH_FN void SWEReconstructPrimitiveWithJacobian(const PetscReal cons[3], PetscReal tiny_h, PetscReal h_anuga, PetscReal q[3],
                                                          PetscReal P[3][3]) {
  PetscReal h = cons[0], hu = cons[1], hv = cons[2];

  for (PetscInt i = 0; i < 3; ++i)
    for (PetscInt j = 0; j < 3; ++j) P[i][j] = 0.0;
  P[0][0] = 1.0;

  q[0] = h;
  if (h < tiny_h) {
    q[1] = 0.0;
    q[2] = 0.0;
    // dry: velocities are identically zero; their derivatives vanish
  } else {
    PetscReal D  = Square(h) + Square(h_anuga);
    PetscReal D2 = Square(D);
    q[1]         = hu * h / D;
    q[2]         = hv * h / D;
    // d(x h / D)/dh = x (D - 2 h^2) / D^2 = x (h_anuga^2 - h^2) / D^2
    P[1][0] = hu * (Square(h_anuga) - Square(h)) / D2;
    P[1][1] = h / D;
    P[2][0] = hv * (Square(h_anuga) - Square(h)) / D2;
    P[2][2] = h / D;
  }
}

// Computes B = dF_phys/dq for the physical (rotated) flux
//   F_phys(q) = { h uperp, h u uperp + g h^2 cn / 2, h v uperp + g h^2 sn / 2 },
// with uperp = u cn + v sn, as assembled in ComputeSWERoeFlux().
RDY_MATH_FN void SWEPhysicalFluxJacobianPrim(const PetscReal q[3], PetscReal sn, PetscReal cn, PetscReal B[3][3]) {
  PetscReal h = q[0], u = q[1], v = q[2];
  PetscReal uperp = u * cn + v * sn;

  B[0][0] = uperp;
  B[0][1] = h * cn;
  B[0][2] = h * sn;

  B[1][0] = u * uperp + GRAVITY * h * cn;
  B[1][1] = h * (uperp + u * cn);
  B[1][2] = h * u * sn;

  B[2][0] = v * uperp + GRAVITY * h * sn;
  B[2][1] = h * v * cn;
  B[2][2] = h * (uperp + v * sn);
}

// Exact directional derivative of the Roe interface flux in PRIMITIVE
// variables: given (qL, qR) and perturbation directions (dqL, dqR), computes
// dF[3], the differential of the flux assembled in ComputeSWERoeFlux(). This
// is hand-written forward-mode differentiation mirroring
// ComputeSWERoeEigenspectrum() and the flux assembly line by line; at |.|
// kinks and entropy-fix branch switches the code's own branch is followed
// (consistent one-sided derivative).
RDY_MATH_FN void SWERoeFluxDifferentialPrim(const PetscReal qL[3], const PetscReal qR[3], PetscReal sn, PetscReal cn,
                                                 const PetscReal dqL[3], const PetscReal dqR[3], PetscReal dF[3]) {
  PetscReal hl = qL[0], ul = qL[1], vl = qL[2];
  PetscReal hr = qR[0], ur = qR[1], vr = qR[2];
  PetscReal dhl = dqL[0], dul = dqL[1], dvl = dqL[2];
  PetscReal dhr = dqR[0], dur = dqR[1], dvr = dqR[2];

  // --- Roe averages (mirrors ComputeSWERoeEigenspectrum) ---
  PetscReal duml = PetscSqrtReal(hl), dumr = PetscSqrtReal(hr);
  PetscReal d_duml = (duml > 0.0) ? dhl / (2.0 * duml) : 0.0;
  PetscReal d_dumr = (dumr > 0.0) ? dhr / (2.0 * dumr) : 0.0;

  PetscReal cl = PetscSqrtReal(GRAVITY * hl), cr = PetscSqrtReal(GRAVITY * hr);
  PetscReal d_cl = (cl > 0.0) ? GRAVITY * dhl / (2.0 * cl) : 0.0;
  PetscReal d_cr = (cr > 0.0) ? GRAVITY * dhr / (2.0 * cr) : 0.0;

  PetscReal hhat   = duml * dumr;
  PetscReal d_hhat = d_duml * dumr + duml * d_dumr;

  PetscReal ssum   = duml + dumr;
  PetscReal d_ssum = d_duml + d_dumr;
  PetscReal uhat   = (duml * ul + dumr * ur) / ssum;
  PetscReal d_uhat = ((d_duml * ul + duml * dul + d_dumr * ur + dumr * dur) - uhat * d_ssum) / ssum;
  PetscReal vhat   = (duml * vl + dumr * vr) / ssum;
  PetscReal d_vhat = ((d_duml * vl + duml * dvl + d_dumr * vr + dumr * dvr) - vhat * d_ssum) / ssum;

  PetscReal chat   = PetscSqrtReal(0.5 * GRAVITY * (hl + hr));
  PetscReal d_chat = (chat > 0.0) ? 0.25 * GRAVITY * (dhl + dhr) / chat : 0.0;

  PetscReal uperp   = uhat * cn + vhat * sn;
  PetscReal d_uperp = d_uhat * cn + d_vhat * sn;

  // --- differences ---
  PetscReal dh = hr - hl, du = ur - ul, dv = vr - vl;
  PetscReal d_dh = dhr - dhl, d_du = dur - dul, d_dv = dvr - dvl;
  PetscReal dupar = -du * sn + dv * cn, duperp = du * cn + dv * sn;
  PetscReal d_dupar = -d_du * sn + d_dv * cn, d_duperp = d_du * cn + d_dv * sn;

  // --- eigenvectors R (columns 0 and 2 depend on the averages) ---
  PetscReal R[3][3] = {{1.0, 0.0, 1.0}, {uhat - chat * cn, -sn, uhat + chat * cn}, {vhat - chat * sn, cn, vhat + chat * sn}};
  PetscReal dR[3][3] = {{0.0, 0.0, 0.0},
                        {d_uhat - d_chat * cn, 0.0, d_uhat + d_chat * cn},
                        {d_vhat - d_chat * sn, 0.0, d_vhat + d_chat * sn}};

  // --- eigenvalues with the critical-flow (entropy) fix, branch-faithful ---
  PetscReal uperpl = ul * cn + vl * sn, uperpr = ur * cn + vr * sn;
  PetscReal d_uperpl = dul * cn + dvl * sn, d_uperpr = dur * cn + dvr * sn;

  PetscReal a1   = PetscAbsReal(uperp - chat);
  PetscReal s1   = (uperp - chat >= 0.0) ? 1.0 : -1.0;
  PetscReal d_a1 = s1 * (d_uperp - d_chat);
  PetscReal a2   = PetscAbsReal(uperp);
  PetscReal s2   = (uperp >= 0.0) ? 1.0 : -1.0;
  PetscReal d_a2 = s2 * d_uperp;
  PetscReal a3   = PetscAbsReal(uperp + chat);
  PetscReal s3   = (uperp + chat >= 0.0) ? 1.0 : -1.0;
  PetscReal d_a3 = s3 * (d_uperp + d_chat);

  PetscReal al1 = uperpl - cl, ar1 = uperpr - cr;
  PetscReal d_al1 = d_uperpl - d_cl, d_ar1 = d_uperpr - d_cr;
  PetscReal da1_raw = 2.0 * (ar1 - al1);
  PetscReal da1     = PetscMax(0.0, da1_raw);
  PetscReal d_da1   = (da1_raw > 0.0) ? 2.0 * (d_ar1 - d_al1) : 0.0;

  PetscReal lam1, d_lam1;
  if (a1 < da1) {
    lam1   = 0.5 * (a1 * a1 / da1 + da1);
    d_lam1 = (a1 / da1) * d_a1 + 0.5 * (1.0 - a1 * a1 / (da1 * da1)) * d_da1;
  } else {
    lam1   = a1;
    d_lam1 = d_a1;
  }

  PetscReal al3 = uperpl + cl, ar3 = uperpr + cr;
  PetscReal d_al3 = d_uperpl + d_cl, d_ar3 = d_uperpr + d_cr;
  PetscReal da3_raw = 2.0 * (ar3 - al3);
  PetscReal da3     = PetscMax(0.0, da3_raw);
  PetscReal d_da3   = (da3_raw > 0.0) ? 2.0 * (d_ar3 - d_al3) : 0.0;

  PetscReal lam3, d_lam3;
  if (a3 < da3) {
    lam3   = 0.5 * (a3 * a3 / da3 + da3);
    d_lam3 = (a3 / da3) * d_a3 + 0.5 * (1.0 - a3 * a3 / (da3 * da3)) * d_da3;
  } else {
    lam3   = a3;
    d_lam3 = d_a3;
  }

  PetscReal lam[3]   = {lam1, a2, lam3};
  PetscReal d_lam[3] = {d_lam1, d_a2, d_lam3};

  // --- wave strengths dW ---
  PetscReal hoc   = (chat > 0.0) ? hhat / chat : 0.0;
  PetscReal d_hoc = (chat > 0.0) ? (d_hhat - hoc * d_chat) / chat : 0.0;
  PetscReal W[3]   = {0.5 * (dh - hoc * duperp), hhat * dupar, 0.5 * (dh + hoc * duperp)};
  PetscReal d_W[3] = {0.5 * (d_dh - d_hoc * duperp - hoc * d_duperp), d_hhat * dupar + hhat * d_dupar,
                      0.5 * (d_dh + d_hoc * duperp + hoc * d_duperp)};

  // --- physical fluxes ---
  PetscReal FLv[3] = {uperpl * hl, ul * uperpl * hl + 0.5 * GRAVITY * hl * hl * cn, vl * uperpl * hl + 0.5 * GRAVITY * hl * hl * sn};
  PetscReal FRv[3] = {uperpr * hr, ur * uperpr * hr + 0.5 * GRAVITY * hr * hr * cn, vr * uperpr * hr + 0.5 * GRAVITY * hr * hr * sn};
  (void)FLv;
  (void)FRv;
  PetscReal d_FL[3] = {d_uperpl * hl + uperpl * dhl,
                       dul * uperpl * hl + ul * d_uperpl * hl + ul * uperpl * dhl + GRAVITY * hl * dhl * cn,
                       dvl * uperpl * hl + vl * d_uperpl * hl + vl * uperpl * dhl + GRAVITY * hl * dhl * sn};
  PetscReal d_FR[3] = {d_uperpr * hr + uperpr * dhr,
                       dur * uperpr * hr + ur * d_uperpr * hr + ur * uperpr * dhr + GRAVITY * hr * dhr * cn,
                       dvr * uperpr * hr + vr * d_uperpr * hr + vr * uperpr * dhr + GRAVITY * hr * dhr * sn};

  // --- assemble dF = 1/2 (dFL + dFR - d(R Lam W)) ---
  for (PetscInt i = 0; i < 3; ++i) {
    PetscReal d_diss = 0.0;
    for (PetscInt k = 0; k < 3; ++k) {
      d_diss += dR[i][k] * lam[k] * W[k] + R[i][k] * d_lam[k] * W[k] + R[i][k] * lam[k] * d_W[k];
    }
    dF[i] = 0.5 * (d_FL[i] + d_FR[i] - d_diss);
  }
}

// Computes the two 3x3 Jacobian blocks of the Roe interface flux with respect
// to the conservative left and right states:
//   dFdUL = dF/d(uL), dFdUR = dF/d(uR)   (exact; see header note).
// Assembled column-by-column from the exact primitive-space differential,
// chained through the primitive-reconstruction Jacobians P_L, P_R.
RDY_MATH_FN void SWERoeFluxJacobian(const PetscReal consL[3], const PetscReal consR[3], PetscReal sn, PetscReal cn, PetscReal tiny_h,
                                         PetscReal h_anuga, PetscReal dFdUL[3][3], PetscReal dFdUR[3][3]) {

  // both sides dry: no flux, no dependence (the RHS skips these edges)
  if (consL[0] < tiny_h && consR[0] < tiny_h) {
    for (PetscInt i = 0; i < 3; ++i)
      for (PetscInt j = 0; j < 3; ++j) {
        dFdUL[i][j] = 0.0;
        dFdUR[i][j] = 0.0;
      }
    return;
  }

  PetscReal qL[3], qR[3], PL[3][3], PR[3][3];
  SWEReconstructPrimitiveWithJacobian(consL, tiny_h, h_anuga, qL, PL);
  SWEReconstructPrimitiveWithJacobian(consR, tiny_h, h_anuga, qR, PR);

  const PetscReal zero[3] = {0.0, 0.0, 0.0};
  for (PetscInt j = 0; j < 3; ++j) {
    // primitive-space directions = columns of P (chain rule through the
    // reconstruction), so each dF is directly a conservative-space column
    PetscReal dirL[3] = {PL[0][j], PL[1][j], PL[2][j]};
    PetscReal dirR[3] = {PR[0][j], PR[1][j], PR[2][j]};
    PetscReal dFL[3], dFR[3];
    SWERoeFluxDifferentialPrim(qL, qR, sn, cn, dirL, zero, dFL);
    SWERoeFluxDifferentialPrim(qL, qR, sn, cn, zero, dirR, dFR);
    for (PetscInt i = 0; i < 3; ++i) {
      dFdUL[i][j] = dFL[i];
      dFdUR[i][j] = dFR[i];
    }
  }
}

// Computes the 3x3 diagonal Jacobian block D = dS/d(h, hu, hv) of the
// EXPLICIT source treatment (ApplySourceExplicit in swe_petsc.c):
//   S = ( s_ext0,  -g dzdx h - tbx + s_ext1,  -g dzdy h - tby + s_ext2 ),
//   tb = g n^2 h^{-7/3} m,  m = |(hu, hv)|,  tbx = tb hu,  tby = tb hv,
// with the friction terms zero for h < tiny_h (and the external sources
// independent of the state). At m = 0 the friction is O(m^2), so its
// derivative is zero there (the consistent one-sided value).
// Friction-only part: D_f = d(S_fric)/d(h, hu, hv) with
// S_fric = (0, -tbx, -tby), tb = g n^2 h^{-7/3} m, m = |(hu, hv)|.
RDY_MATH_FN void SWEFrictionJacobian(const PetscReal cons[3], PetscReal n_manning, PetscReal tiny_h, PetscReal D[3][3]) {
  PetscReal h = cons[0], hu = cons[1], hv = cons[2];

  for (PetscInt i = 0; i < 3; ++i)
    for (PetscInt j = 0; j < 3; ++j) D[i][j] = 0.0;

  if (h >= tiny_h) {
    PetscReal m = PetscSqrtReal(Square(hu) + Square(hv));
    if (m > 0.0) {
      PetscReal c = GRAVITY * Square(n_manning) * PetscPowReal(h, -7.0 / 3.0);
      // d(tbx)/dh = -(7/3) c m hu / h, etc.; rows carry -d(tb*)/d(.)
      D[1][0] = (7.0 / 3.0) * c * m * hu / h;
      D[1][1] = -c * (m + Square(hu) / m);
      D[1][2] = -c * hu * hv / m;
      D[2][0] = (7.0 / 3.0) * c * m * hv / h;
      D[2][1] = -c * hu * hv / m;
      D[2][2] = -c * (m + Square(hv) / m);
    }
  }
}

RDY_MATH_FN void SWESourceJacobian(const PetscReal cons[3], PetscReal n_manning, PetscReal dzdx, PetscReal dzdy, PetscReal tiny_h,
                                        PetscReal D[3][3]) {
  SWEFrictionJacobian(cons, n_manning, tiny_h, D);

  // bed slope: -g dz/dx h contributes to the h column of the momentum rows
  D[1][0] += -GRAVITY * dzdx;
  D[2][0] += -GRAVITY * dzdy;
}

#pragma GCC diagnostic pop
#pragma clang diagnostic pop

#endif  // SWE_ROE_FLUX_JACOBIAN_PETSC_H
