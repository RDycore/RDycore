#ifndef SWE_ROE_FLUX_JACOBIAN_CEED_H
#define SWE_ROE_FLUX_JACOBIAN_CEED_H

// Analytic Jacobian of the Roe flux computed by SWERiemannFlux_Roe (swe_roe_flux_ceed.h).
//
// The Jacobian is the exact derivative of the flux *as coded*: the chain rule of
// paper Section 6 (dF/dU = d(F_L + F_R)/dU - d(R |Lambda| dW)/dU), applied to every
// intermediate of the kernel, including the velocity desingularization
// u = hu h / (h^2 + h_anuga^2), the tiny_h dry switch, and the critical-flow entropy
// fix of |lambda_1| and |lambda_3|. The chain rule is carried out in forward mode:
// every intermediate carries its value and its derivatives with respect to the six
// inputs (h, hu, hv) of the left and right states. At non-differentiable points
// (|x| at x = 0, max(0, x), the entropy-fix switch) the derivative of the active
// branch is used.
//
// This header is shared by the PETSc and CEED backends, and is GPU-JIT safe
// (no libc beyond math functions known to the JIT compiler).

#include <ceed/types.h>

#include "swe_types_ceed.h"

#pragma GCC diagnostic   push
#pragma GCC diagnostic   ignored "-Wimplicit-function-declaration"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-function-declaration"

#ifndef Square
#define Square(x) ((x) * (x))
#endif

// number of independent variables: (h, hu, hv) on the left and right of an edge
#define SWE_JAC_NDIR 6

// a value together with its derivatives with respect to the SWE_JAC_NDIR inputs
typedef struct {
  CeedScalar v;
  CeedScalar d[SWE_JAC_NDIR];
} SWEDual;

CEED_QFUNCTION_HELPER SWEDual SWEDualConst(CeedScalar v) {
  SWEDual r;
  r.v = v;
  for (CeedInt k = 0; k < SWE_JAC_NDIR; k++) r.d[k] = 0.0;
  return r;
}

CEED_QFUNCTION_HELPER SWEDual SWEDualVar(CeedScalar v, CeedInt dir) {
  SWEDual r = SWEDualConst(v);
  r.d[dir]  = 1.0;
  return r;
}

CEED_QFUNCTION_HELPER SWEDual SWEDualAdd(SWEDual a, SWEDual b) {
  SWEDual r;
  r.v = a.v + b.v;
  for (CeedInt k = 0; k < SWE_JAC_NDIR; k++) r.d[k] = a.d[k] + b.d[k];
  return r;
}

CEED_QFUNCTION_HELPER SWEDual SWEDualSub(SWEDual a, SWEDual b) {
  SWEDual r;
  r.v = a.v - b.v;
  for (CeedInt k = 0; k < SWE_JAC_NDIR; k++) r.d[k] = a.d[k] - b.d[k];
  return r;
}

CEED_QFUNCTION_HELPER SWEDual SWEDualScale(CeedScalar s, SWEDual a) {
  SWEDual r;
  r.v = s * a.v;
  for (CeedInt k = 0; k < SWE_JAC_NDIR; k++) r.d[k] = s * a.d[k];
  return r;
}

CEED_QFUNCTION_HELPER SWEDual SWEDualMul(SWEDual a, SWEDual b) {
  SWEDual r;
  r.v = a.v * b.v;
  for (CeedInt k = 0; k < SWE_JAC_NDIR; k++) r.d[k] = a.d[k] * b.v + a.v * b.d[k];
  return r;
}

CEED_QFUNCTION_HELPER SWEDual SWEDualDiv(SWEDual a, SWEDual b) {
  SWEDual          r;
  const CeedScalar inv = 1.0 / b.v;
  r.v                  = a.v * inv;
  for (CeedInt k = 0; k < SWE_JAC_NDIR; k++) r.d[k] = (a.d[k] - r.v * b.d[k]) * inv;
  return r;
}

// sqrt with a zero derivative at a = 0 (a dry state has no sensitivity here)
CEED_QFUNCTION_HELPER SWEDual SWEDualSqrt(SWEDual a) {
  SWEDual r;
  r.v                  = sqrt(a.v);
  const CeedScalar fac = (r.v > 0.0) ? 0.5 / r.v : 0.0;
  for (CeedInt k = 0; k < SWE_JAC_NDIR; k++) r.d[k] = fac * a.d[k];
  return r;
}

// |a|, matching fabs(): the derivative of the a >= 0 branch is used at a = 0
CEED_QFUNCTION_HELPER SWEDual SWEDualAbs(SWEDual a) { return (a.v < 0.0) ? SWEDualScale(-1.0, a) : a; }

// max(0, a), matching fmax(0.0, a)
CEED_QFUNCTION_HELPER SWEDual SWEDualMax0(SWEDual a) { return (a.v > 0.0) ? a : SWEDualConst(0.0); }

// critical-flow (entropy) fix applied to an eigenvalue magnitude a with threshold da,
// matching ComputeSWEEigenspectrum_Roe: if (a < da) a = 0.5 * (a*a/da + da)
CEED_QFUNCTION_HELPER SWEDual SWEDualEntropyFix(SWEDual a, SWEDual da) {
  if (a.v < da.v) {
    return SWEDualScale(0.5, SWEDualAdd(SWEDualDiv(SWEDualMul(a, a), da), da));
  }
  return a;
}

// desingularized velocity u = hu * h / (h^2 + h_anuga^2), zero for h <= tiny_h (SafeDiv)
CEED_QFUNCTION_HELPER SWEDual SWEDualVelocity(SWEDual h, SWEDual hu, CeedScalar tiny_h, CeedScalar h_anuga) {
  if (h.v > tiny_h) {
    SWEDual denom = SWEDualAdd(SWEDualMul(h, h), SWEDualConst(Square(h_anuga)));
    return SWEDualDiv(SWEDualMul(hu, h), denom);
  }
  return SWEDualConst(0.0);
}

/// Roe flux of SWERiemannFlux_Roe (with dhv = 0) in terms of depths and velocities given as dual
/// numbers (the part of the kernel after the velocities are formed). Used directly when a caller
/// supplies velocities (the PETSc hydrostatic-reconstruction residual), and by SWEDualRoeFlux.
CEED_QFUNCTION_HELPER void SWEDualRoeFluxPrimitive(const CeedScalar gravity, SWEDual hl, SWEDual ul, SWEDual vl, SWEDual hr, SWEDual ur, SWEDual vr,
                                                   CeedScalar sn, CeedScalar cn, SWEDual F[3], CeedScalar *amax) {
  // Roe averages (ComputeSWEEigenspectrum_Roe)
  const CeedScalar sqrt_gravity = sqrt(gravity);
  const SWEDual    duml         = SWEDualSqrt(hl);
  const SWEDual    dumr         = SWEDualSqrt(hr);
  const SWEDual    cl           = SWEDualScale(sqrt_gravity, duml);
  const SWEDual    cr           = SWEDualScale(sqrt_gravity, dumr);
  const SWEDual    hhat         = SWEDualMul(duml, dumr);
  const SWEDual    dsum         = SWEDualAdd(duml, dumr);
  const SWEDual    uhat         = SWEDualDiv(SWEDualAdd(SWEDualMul(duml, ul), SWEDualMul(dumr, ur)), dsum);
  const SWEDual    vhat         = SWEDualDiv(SWEDualAdd(SWEDualMul(duml, vl), SWEDualMul(dumr, vr)), dsum);
  const SWEDual    chat         = SWEDualSqrt(SWEDualScale(0.5 * gravity, SWEDualAdd(hl, hr)));
  const SWEDual    uperp        = SWEDualAdd(SWEDualScale(cn, uhat), SWEDualScale(sn, vhat));

  const SWEDual dh     = SWEDualSub(hr, hl);
  const SWEDual du     = SWEDualSub(ur, ul);
  const SWEDual dv     = SWEDualSub(vr, vl);
  const SWEDual dupar  = SWEDualAdd(SWEDualScale(-sn, du), SWEDualScale(cn, dv));
  const SWEDual duperp = SWEDualAdd(SWEDualScale(cn, du), SWEDualScale(sn, dv));

  // right eigenvectors (constant entries are handled inline below)
  const SWEDual R10 = SWEDualSub(uhat, SWEDualScale(cn, chat));
  const SWEDual R12 = SWEDualAdd(uhat, SWEDualScale(cn, chat));
  const SWEDual R20 = SWEDualSub(vhat, SWEDualScale(sn, chat));
  const SWEDual R22 = SWEDualAdd(vhat, SWEDualScale(sn, chat));

  // eigenvalue magnitudes with the critical-flow fix
  const SWEDual uperpl = SWEDualAdd(SWEDualScale(cn, ul), SWEDualScale(sn, vl));
  const SWEDual uperpr = SWEDualAdd(SWEDualScale(cn, ur), SWEDualScale(sn, vr));
  SWEDual       a1     = SWEDualAbs(SWEDualSub(uperp, chat));
  const SWEDual a2     = SWEDualAbs(uperp);
  SWEDual       a3     = SWEDualAbs(SWEDualAdd(uperp, chat));

  const SWEDual al1 = SWEDualSub(uperpl, cl);
  const SWEDual ar1 = SWEDualSub(uperpr, cr);
  const SWEDual da1 = SWEDualMax0(SWEDualScale(2.0, SWEDualSub(ar1, al1)));
  a1                = SWEDualEntropyFix(a1, da1);
  const SWEDual al3 = SWEDualAdd(uperpl, cl);
  const SWEDual ar3 = SWEDualAdd(uperpr, cr);
  const SWEDual da3 = SWEDualMax0(SWEDualScale(2.0, SWEDualSub(ar3, al3)));
  a3                = SWEDualEntropyFix(a3, da3);

  // wave strengths
  const SWEDual hd  = SWEDualDiv(SWEDualMul(hhat, duperp), chat);
  const SWEDual dW0 = SWEDualScale(0.5, SWEDualSub(dh, hd));
  const SWEDual dW1 = SWEDualMul(hhat, dupar);
  const SWEDual dW2 = SWEDualScale(0.5, SWEDualAdd(dh, hd));

  // physical fluxes
  const SWEDual qperpl = SWEDualMul(uperpl, hl);
  const SWEDual qperpr = SWEDualMul(uperpr, hr);
  const SWEDual pl     = SWEDualScale(0.5 * gravity, SWEDualMul(hl, hl));
  const SWEDual pr     = SWEDualScale(0.5 * gravity, SWEDualMul(hr, hr));
  const SWEDual FL0 = qperpl, FL1 = SWEDualAdd(SWEDualMul(ul, qperpl), SWEDualScale(cn, pl)),
                FL2 = SWEDualAdd(SWEDualMul(vl, qperpl), SWEDualScale(sn, pl));
  const SWEDual FR0 = qperpr, FR1 = SWEDualAdd(SWEDualMul(ur, qperpr), SWEDualScale(cn, pr)),
                FR2 = SWEDualAdd(SWEDualMul(vr, qperpr), SWEDualScale(sn, pr));

  // dissipation B = R |Lambda| dW, with R[0] = (1, 0, 1), R[1][1] = -sn, R[2][1] = cn
  const SWEDual w0 = SWEDualMul(a1, dW0);
  const SWEDual w1 = SWEDualMul(a2, dW1);
  const SWEDual w2 = SWEDualMul(a3, dW2);
  const SWEDual B0 = SWEDualAdd(w0, w2);
  const SWEDual B1 = SWEDualAdd(SWEDualAdd(SWEDualMul(R10, w0), SWEDualScale(-sn, w1)), SWEDualMul(R12, w2));
  const SWEDual B2 = SWEDualAdd(SWEDualAdd(SWEDualMul(R20, w0), SWEDualScale(cn, w1)), SWEDualMul(R22, w2));

  F[0] = SWEDualScale(0.5, SWEDualSub(SWEDualAdd(FL0, FR0), B0));
  F[1] = SWEDualScale(0.5, SWEDualSub(SWEDualAdd(FL1, FR1), B1));
  F[2] = SWEDualScale(0.5, SWEDualSub(SWEDualAdd(FL2, FR2), B2));

  *amax = chat.v + fabs(uperp.v);
}

/// Roe flux of SWERiemannFlux_Roe (with dhv = 0, i.e. no BS2002 well balancing) for conservative
/// states given as dual numbers, so that callers can chain the Roe flux with a state reconstruction
/// (e.g. hydrostatic reconstruction). Returns F (with derivatives) and the maximum wave speed.
CEED_QFUNCTION_HELPER void SWEDualRoeFlux(const CeedScalar gravity, const CeedScalar tiny_h, const CeedScalar h_anuga, SWEDual hl, SWEDual hul,
                                          SWEDual hvl, SWEDual hr, SWEDual hur, SWEDual hvr, CeedScalar sn, CeedScalar cn, SWEDual F[3],
                                          CeedScalar *amax) {
  const SWEDual ul = SWEDualVelocity(hl, hul, tiny_h, h_anuga);
  const SWEDual vl = SWEDualVelocity(hl, hvl, tiny_h, h_anuga);
  const SWEDual ur = SWEDualVelocity(hr, hur, tiny_h, h_anuga);
  const SWEDual vr = SWEDualVelocity(hr, hvr, tiny_h, h_anuga);
  SWEDualRoeFluxPrimitive(gravity, hl, ul, vl, hr, ur, vr, sn, cn, F, amax);
}

/// Computes the Roe flux of SWERiemannFlux_Roe (with dhv = 0, i.e. no BS2002 well balancing)
/// and its Jacobians with respect to the left and right conservative states.
/// @param [in]  gravity  acceleration due to gravity
/// @param [in]  tiny_h   water height at or below which a state is dry
/// @param [in]  h_anuga  ANUGA velocity desingularization height
/// @param [in]  qL, qR   left and right states (h, hu, hv)
/// @param [in]  sn, cn   sine and cosine of the edge normal angle
/// @param [out] flux     the Roe flux (3 components)
/// @param [out] dF_dqL   dF_i/dqL_j
/// @param [out] dF_dqR   dF_i/dqR_j
/// @param [out] amax     maximum wave speed
CEED_QFUNCTION_HELPER void SWERiemannFluxJacobian_Roe(const CeedScalar gravity, const CeedScalar tiny_h, const CeedScalar h_anuga, SWEState qL,
                                                      SWEState qR, CeedScalar sn, CeedScalar cn, CeedScalar flux[3], CeedScalar dF_dqL[3][3],
                                                      CeedScalar dF_dqR[3][3], CeedScalar *amax) {
  SWEDual F[3];
  SWEDualRoeFlux(gravity, tiny_h, h_anuga, SWEDualVar(qL.h, 0), SWEDualVar(qL.hu, 1), SWEDualVar(qL.hv, 2), SWEDualVar(qR.h, 3), SWEDualVar(qR.hu, 4),
                 SWEDualVar(qR.hv, 5), sn, cn, F, amax);
  for (CeedInt i = 0; i < 3; i++) {
    flux[i] = F[i].v;
    for (CeedInt j = 0; j < 3; j++) {
      dF_dqL[i][j] = F[i].d[j];
      dF_dqR[i][j] = F[i].d[3 + j];
    }
  }
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop

#endif
