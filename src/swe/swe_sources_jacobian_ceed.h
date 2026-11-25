#ifndef SWE_SOURCES_JACOBIAN_CEED_H
#define SWE_SOURCES_JACOBIAN_CEED_H

// Analytic Jacobians of the pointwise SWE source terms (bed slope and fully implicit
// bed friction, see SWEFullyImplicitBedFrictionRoughness) and of the boundary-state maps
// U_R(U_L) used by the reflecting and critical-outflow boundary conditions (paper
// Sections 7.4 and 7.5). Shared by the PETSc and CEED backends; GPU-JIT safe.

#include "swe_roe_flux_jacobian_ceed.h"

#pragma GCC diagnostic   push
#pragma GCC diagnostic   ignored "-Wimplicit-function-declaration"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-function-declaration"

/// Jacobian of the cell source term S(U) = [0, -g h dz/dx - tbx, -g h dz/dy - tby] with
/// tbx = hu * g n^2 h^{-4/3} |u| (desingularized velocity, zero for h <= tiny_h).
/// @param [out] dS_dq  dS_i/dq_j
CEED_QFUNCTION_HELPER void SWESourceJacobian(const CeedScalar gravity, const CeedScalar tiny_h, const CeedScalar h_anuga, SWEState q,
                                             CeedScalar dz_dx, CeedScalar dz_dy, CeedScalar mannings_n, CeedScalar dS_dq[3][3]) {
  for (CeedInt i = 0; i < 3; i++)
    for (CeedInt j = 0; j < 3; j++) dS_dq[i][j] = 0.0;

  // bed slope: S1 = -g h dz/dx, S2 = -g h dz/dy
  dS_dq[1][0] = -gravity * dz_dx;
  dS_dq[2][0] = -gravity * dz_dy;

  if (q.h > tiny_h) {
    const SWEDual h  = SWEDualVar(q.h, 0);
    const SWEDual hu = SWEDualVar(q.hu, 1);
    const SWEDual hv = SWEDualVar(q.hv, 2);
    const SWEDual u  = SWEDualVelocity(h, hu, tiny_h, h_anuga);
    const SWEDual v  = SWEDualVelocity(h, hv, tiny_h, h_anuga);

    // tb = g n^2 h^{-1/3} |u| / h = g n^2 h^{-4/3} |u|
    const SWEDual speed = SWEDualSqrt(SWEDualAdd(SWEDualMul(u, u), SWEDualMul(v, v)));
    SWEDual       hpow  = SWEDualConst(pow(q.h, -4.0 / 3.0));
    hpow.d[0]           = (-4.0 / 3.0) * pow(q.h, -7.0 / 3.0);
    const SWEDual tb    = SWEDualScale(gravity * Square(mannings_n), SWEDualMul(hpow, speed));
    const SWEDual tbx   = SWEDualMul(hu, tb);
    const SWEDual tby   = SWEDualMul(hv, tb);
    for (CeedInt j = 0; j < 3; j++) {
      dS_dq[1][j] -= tbx.d[j];
      dS_dq[2][j] -= tby.d[j];
    }
  }
}

/// Reflecting wall: U_R = M U_L, returns M = dU_R/dU_L (paper Section 7.5.1)
CEED_QFUNCTION_HELPER void SWEReflectingStateJacobian(CeedScalar sn, CeedScalar cn, CeedScalar dqR_dqL[3][3]) {
  const CeedScalar dum1 = sn * sn - cn * cn;
  const CeedScalar dum2 = 2.0 * sn * cn;
  dqR_dqL[0][0]         = 1.0;
  dqR_dqL[0][1]         = 0.0;
  dqR_dqL[0][2]         = 0.0;
  dqR_dqL[1][0]         = 0.0;
  dqR_dqL[1][1]         = dum1;
  dqR_dqL[1][2]         = -dum2;
  dqR_dqL[2][0]         = 0.0;
  dqR_dqL[2][1]         = -dum2;
  dqR_dqL[2][2]         = -dum1;
}

/// Critical outflow: U_R = ((q^2/g)^{1/3}, |q| cn, |q| sn) with q = hu cn + hv sn, matching
/// SWEBoundaryFlux_Outflow (which forms hR*sqrt(g hR) = |q|). Returns dU_R/dU_L
/// (paper Section 7.5.2) with a floor on |q| for the |q|^{-1/3} singularity.
CEED_QFUNCTION_HELPER void SWECriticalOutflowStateJacobian(const CeedScalar gravity, SWEState qL, CeedScalar sn, CeedScalar cn,
                                                           CeedScalar dqR_dqL[3][3]) {
  const CeedScalar q    = qL.hu * cn + qL.hv * sn;
  const CeedScalar sgn  = (q < 0.0) ? -1.0 : 1.0;
  const CeedScalar qabs = fmax(fabs(q), 1.0e-12);
  const CeedScalar hR   = pow(qabs * qabs / gravity, 1.0 / 3.0);
  const CeedScalar dhdq = 2.0 * hR / (3.0 * qabs) * sgn;
  // d(hR sqrt(g hR))/dq = d(sqrt(g) hR^{3/2})/dq = 1.5 sqrt(g hR) dhR/dq
  const CeedScalar dmdq = 1.5 * sqrt(gravity * hR) * dhdq;
  dqR_dqL[0][0]         = 0.0;
  dqR_dqL[0][1]         = dhdq * cn;
  dqR_dqL[0][2]         = dhdq * sn;
  dqR_dqL[1][0]         = 0.0;
  dqR_dqL[1][1]         = dmdq * cn * cn;
  dqR_dqL[1][2]         = dmdq * cn * sn;
  dqR_dqL[2][0]         = 0.0;
  dqR_dqL[2][1]         = dmdq * sn * cn;
  dqR_dqL[2][2]         = dmdq * sn * sn;
}

/// Total derivative of a boundary flux with respect to the interior state:
/// dF/dU_L + dF/dU_R * dU_R/dU_L (paper Section 7.5)
CEED_QFUNCTION_HELPER void SWEBoundaryFluxTotalJacobian(CeedScalar dF_dqL[3][3], CeedScalar dF_dqR[3][3], CeedScalar dqR_dqL[3][3],
                                                        CeedScalar dF_total[3][3]) {
  for (CeedInt i = 0; i < 3; i++) {
    for (CeedInt j = 0; j < 3; j++) {
      CeedScalar sum = dF_dqL[i][j];
      for (CeedInt k = 0; k < 3; k++) sum += dF_dqR[i][k] * dqR_dqL[k][j];
      dF_total[i][j] = sum;
    }
  }
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop

#endif
