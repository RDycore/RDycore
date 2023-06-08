#ifndef swe_flux_h
#define swe_flux_h

#include <ceed/types.h>

#define SafeDiv(a, b, tiny) ((b) > (tiny) ? (a) / (b) : 0.0)

struct SWEState_ {
  CeedScalar h, hu, hv;
};
typedef struct SWEState_ SWEState;

CEED_QFUNCTION_HELPER void SWERiemannFlux_Roe(SWEState qL, SWEState qR, CeedScalar sn, CeedScalar cn, CeedScalar flux[], CeedScalar *amax) {
  const CeedScalar GRAVITY      = 9.806;
  const CeedScalar SQRT_GRAVITY = sqrt(GRAVITY);
  const CeedScalar tiny_h       = 1e-7;
  const CeedScalar hl = qL.h, hr = qR.h;
  const CeedScalar ul = SafeDiv(qL.hu, hl, tiny_h), vl = SafeDiv(qL.hv, hl, tiny_h);
  const CeedScalar ur = SafeDiv(qR.hu, hr, tiny_h), vr = SafeDiv(qR.hv, hr, tiny_h);
  CeedScalar       duml  = sqrt(hl);
  CeedScalar       dumr  = sqrt(hr);
  CeedScalar       cl    = SQRT_GRAVITY * duml;
  CeedScalar       cr    = SQRT_GRAVITY * dumr;
  CeedScalar       hhat  = duml * dumr;
  CeedScalar       uhat  = (duml * ul + dumr * ur) / (duml + dumr);
  CeedScalar       vhat  = (duml * vl + dumr * vr) / (duml + dumr);
  CeedScalar       chat  = sqrt(0.5 * GRAVITY * (hl + hr));
  CeedScalar       uperp = uhat * cn + vhat * sn;

  CeedScalar dh     = hr - hl;
  CeedScalar du     = ur - ul;
  CeedScalar dv     = vr - vl;
  CeedScalar dupar  = -du * sn + dv * cn;
  CeedScalar duperp = du * cn + dv * sn;

  CeedScalar dW[3];
  dW[0] = 0.5 * (dh - hhat * duperp / chat);
  dW[1] = hhat * dupar;
  dW[2] = 0.5 * (dh + hhat * duperp / chat);

  CeedScalar uperpl = ul * cn + vl * sn;
  CeedScalar uperpr = ur * cn + vr * sn;
  CeedScalar al1    = uperpl - cl;
  CeedScalar al3    = uperpl + cl;
  CeedScalar ar1    = uperpr - cr;
  CeedScalar ar3    = uperpr + cr;

  CeedScalar R[3][3];
  R[0][0] = 1.0;
  R[0][1] = 0.0;
  R[0][2] = 1.0;
  R[1][0] = uhat - chat * cn;
  R[1][1] = -sn;
  R[1][2] = uhat + chat * cn;
  R[2][0] = vhat - chat * sn;
  R[2][1] = cn;
  R[2][2] = vhat + chat * sn;

  CeedScalar da1 = fmax(0.0, 2.0 * (ar1 - al1));
  CeedScalar da3 = fmax(0.0, 2.0 * (ar3 - al3));
  CeedScalar a1  = fabs(uperp - chat);
  CeedScalar a2  = fabs(uperp);
  CeedScalar a3  = fabs(uperp + chat);

  // Critical flow fix
  if (a1 < da1) {
    a1 = 0.5 * (a1 * a1 / da1 + da1);
  }
  if (a3 < da3) {
    a3 = 0.5 * (a3 * a3 / da3 + da3);
  }

  // Compute interface flux
  CeedScalar A[3][3] = {0};
  A[0][0]            = a1;
  A[1][1]            = a2;
  A[2][2]            = a3;

  CeedScalar FL[3], FR[3];
  FL[0] = uperpl * hl;
  FL[1] = ul * uperpl * hl + 0.5 * GRAVITY * hl * hl * cn;
  FL[2] = vl * uperpl * hl + 0.5 * GRAVITY * hl * hl * sn;

  FR[0] = uperpr * hr;
  FR[1] = ur * uperpr * hr + 0.5 * GRAVITY * hr * hr * cn;
  FR[2] = vr * uperpr * hr + 0.5 * GRAVITY * hr * hr * sn;

  // fij = 0.5*(FL + FR - matmul(R,matmul(A,dW))
  flux[0] = 0.5 * (FL[0] + FR[0] - R[0][0] * A[0][0] * dW[0] - R[0][1] * A[1][1] * dW[1] - R[0][2] * A[2][2] * dW[2]);
  flux[1] = 0.5 * (FL[1] + FR[1] - R[1][0] * A[0][0] * dW[0] - R[1][1] * A[1][1] * dW[1] - R[1][2] * A[2][2] * dW[2]);
  flux[2] = 0.5 * (FL[2] + FR[2] - R[2][0] * A[0][0] * dW[0] - R[2][1] * A[1][1] * dW[1] - R[2][2] * A[2][2] * dW[2]);

  *amax = chat + fabs(uperp);
}

CEED_QFUNCTION(SWEFlux_Roe)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const CeedScalar(*geom)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];  // sn, cn, weight_L, weight_R
  const CeedScalar(*q_L)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*q_R)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  CeedScalar(*cell_L)[CEED_Q_VLA]     = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*cell_R)[CEED_Q_VLA]     = (CeedScalar(*)[CEED_Q_VLA])out[1];
  for (CeedInt i = 0; i < Q; i++) {
    SWEState   qL = {q_L[0][i], q_L[1][i], q_L[2][i]};
    SWEState   qR = {q_R[0][i], q_R[1][i], q_R[2][i]};
    CeedScalar flux[3], amax;
    SWERiemannFlux_Roe(qL, qR, geom[0][i], geom[1][i], flux, &amax);
    for (CeedInt j = 0; j < 3; j++) {
      cell_L[j][i] = geom[2][i] * flux[j];
      cell_R[j][i] = geom[3][i] * flux[j];
    }
  }
  return 0;
}

CEED_QFUNCTION(SWEBoundaryFlux_Reflecting_Roe)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const CeedScalar(*geom)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];  // sn, cn, weight_L
  const CeedScalar(*q_L)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  CeedScalar(*cell_L)[CEED_Q_VLA]     = (CeedScalar(*)[CEED_Q_VLA])out[0];
  for (CeedInt i = 0; i < Q; i++) {
    CeedScalar sn = geom[0][i], cn = geom[1][i];
    SWEState   qL   = {q_L[0][i], q_L[1][i], q_L[2][i]};
    CeedScalar dum1 = sn * sn - cn * cn;
    CeedScalar dum2 = 2.0 * sn * cn;
    SWEState   qR   = {qL.h, qL.hu * dum1 - qL.hv * dum2, -qL.hu * dum2 - qL.hv * dum1};
    CeedScalar flux[3], amax;
    SWERiemannFlux_Roe(qL, qR, sn, cn, flux, &amax);
    for (CeedInt j = 0; j < 3; j++) {
      cell_L[j][i] = geom[2][i] * flux[j];
    }
  }
  return 0;
}

CEED_QFUNCTION(SWEBoundaryFlux_Outflow_Roe)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const CeedScalar GRAVITY            = 9.806;
  const CeedScalar(*geom)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];  // sn, cn, weight_L
  const CeedScalar(*q_L)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  CeedScalar(*cell_L)[CEED_Q_VLA]     = (CeedScalar(*)[CEED_Q_VLA])out[0];
  for (CeedInt i = 0; i < Q; i++) {
    CeedScalar sn = geom[0][i], cn = geom[1][i];
    SWEState   qL    = {q_L[0][i], q_L[1][i], q_L[2][i]};
    CeedScalar q     = fabs(qL.hu * cn + qL.hv * sn);
    CeedScalar hR    = pow(q * q / GRAVITY, 1.0 / 3.0);
    CeedScalar speed = sqrt(GRAVITY * hR);
    SWEState   qR    = {hR, hR * speed * cn, hR * speed * sn};
    CeedScalar flux[3], amax;
    SWERiemannFlux_Roe(qL, qR, sn, cn, flux, &amax);
    for (CeedInt j = 0; j < 3; j++) {
      cell_L[j][i] = geom[2][i] * flux[j];
    }
  }
  return 0;
}

#endif  // swe_flux_h
