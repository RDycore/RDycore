#ifndef SWE_ROE_CEED_IMPL_H
#define SWE_ROE_CEED_IMPL_H

#include <ceed/types.h>

// disable compiler warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wimplicit-function-declaration"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-function-declaration"

// computes eigenvalues lambda, right eigenvectors R, parameter change dW, and
// the maximum wave speed for the shallow water equations
CEED_QFUNCTION_HELPER void ComputeSWEEigenspectrum_Roe(const CeedScalar hl, const CeedScalar ul, const CeedScalar vl, const CeedScalar hr,
                                                       const CeedScalar ur, const CeedScalar vr, CeedScalar sn, CeedScalar cn,
                                                       const CeedScalar gravity, CeedScalar lambda[3], CeedScalar R[3][3], CeedScalar dW[3],
                                                       CeedScalar *amax) {
  const CeedScalar sqrt_gravity = sqrt(gravity);

  CeedScalar duml  = sqrt(hl);
  CeedScalar dumr  = sqrt(hr);
  CeedScalar cl    = sqrt_gravity * duml;
  CeedScalar cr    = sqrt_gravity * dumr;
  CeedScalar hhat  = duml * dumr;
  CeedScalar uhat  = (duml * ul + dumr * ur) / (duml + dumr);
  CeedScalar vhat  = (duml * vl + dumr * vr) / (duml + dumr);
  CeedScalar chat  = sqrt(0.5 * gravity * (hl + hr));
  CeedScalar uperp = uhat * cn + vhat * sn;

  CeedScalar dh     = hr - hl;
  CeedScalar du     = ur - ul;
  CeedScalar dv     = vr - vl;
  CeedScalar dupar  = -du * sn + dv * cn;
  CeedScalar duperp = du * cn + dv * sn;

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
  CeedScalar uperpl = ul * cn + vl * sn;
  CeedScalar uperpr = ur * cn + vr * sn;
  CeedScalar a1     = fabs(uperp - chat);
  CeedScalar a2     = fabs(uperp);
  CeedScalar a3     = fabs(uperp + chat);

  // apply critical flow fix
  CeedScalar al1 = uperpl - cl;
  CeedScalar ar1 = uperpr - cr;
  CeedScalar da1 = fmax(0.0, 2.0 * (ar1 - al1));
  if (a1 < da1) {
    a1 = 0.5 * (a1 * a1 / da1 + da1);
  }
  CeedScalar al3 = uperpl + cl;
  CeedScalar ar3 = uperpr + cr;
  CeedScalar da3 = fmax(0.0, 2.0 * (ar3 - al3));
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

// riemann solver -- called by other Q-functions
CEED_QFUNCTION_HELPER void SWERiemannFlux_Roe(const CeedScalar gravity, const CeedScalar tiny_h, const CeedScalar h_anuga, SWEState qL, SWEState qR,
                                              CeedScalar sn, CeedScalar cn, CeedScalar flux[], CeedScalar *amax) {
  const CeedScalar hl = qL.h, hr = qR.h;

  const CeedScalar denom_l = Square(hl) + Square(h_anuga);
  const CeedScalar denom_r = Square(hr) + Square(h_anuga);

  const CeedScalar ul = SafeDiv(qL.hu * hl, denom_l, hl, tiny_h);
  const CeedScalar vl = SafeDiv(qL.hv * hl, denom_l, hl, tiny_h);
  const CeedScalar ur = SafeDiv(qR.hu * hr, denom_r, hr, tiny_h);
  const CeedScalar vr = SafeDiv(qR.hv * hr, denom_r, hr, tiny_h);

  // compute eigenspectrum
  CeedScalar A[3], R[3][3], dW[3];
  ComputeSWEEigenspectrum_Roe(hl, ul, vl, hr, ur, vr, sn, cn, gravity, A, R, dW, amax);

  // compute interface fluxes
  CeedScalar uperpl = ul * cn + vl * sn;
  CeedScalar uperpr = ur * cn + vr * sn;
  CeedScalar FL[3]  = {
       uperpl * hl,
       ul * uperpl * hl + 0.5 * gravity * hl * hl * cn,
       vl * uperpl * hl + 0.5 * gravity * hl * hl * sn,
  };
  CeedScalar FR[3] = {
      uperpr * hr,
      ur * uperpr * hr + 0.5 * gravity * hr * hr * cn,
      vr * uperpr * hr + 0.5 * gravity * hr * hr * sn,
  };

  // fij = 0.5*(FL + FR - matmul(R,matmul(A,dW))
  flux[0] = 0.5 * (FL[0] + FR[0] - R[0][0] * A[0] * dW[0] - R[0][1] * A[1] * dW[1] - R[0][2] * A[2] * dW[2]);
  flux[1] = 0.5 * (FL[1] + FR[1] - R[1][0] * A[0] * dW[0] - R[1][1] * A[1] * dW[1] - R[1][2] * A[2] * dW[2]);
  flux[2] = 0.5 * (FL[2] + FR[2] - R[2][0] * A[0] * dW[0] - R[2][1] * A[1] * dW[1] - R[2][2] * A[2] * dW[2]);
}

#endif
