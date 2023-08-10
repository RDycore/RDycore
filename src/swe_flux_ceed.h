#ifndef swe_flux_ceed_h
#define swe_flux_ceed_h

#include <ceed/types.h>
#include <private/rdycoreimpl.h>

#define SafeDiv(a, b, tiny) ((b) > (tiny) ? (a) / (b) : 0.0)

struct SWEState_ {
  CeedScalar h, hu, hv;
};
typedef struct SWEState_ SWEState;

CEED_QFUNCTION_HELPER void SWERiemannFlux_Roe(const CeedScalar gravity, const CeedScalar tiny_h, SWEState qL, SWEState qR, CeedScalar sn, CeedScalar cn, CeedScalar flux[], CeedScalar *amax) {
  const CeedScalar sqrt_gravity = sqrt(gravity);
  const CeedScalar hl = qL.h, hr = qR.h;

  const CeedScalar ul = SafeDiv(qL.hu, hl, tiny_h), vl = SafeDiv(qL.hv, hl, tiny_h);
  const CeedScalar ur = SafeDiv(qR.hu, hr, tiny_h), vr = SafeDiv(qR.hv, hr, tiny_h);
  CeedScalar       duml  = sqrt(hl);
  CeedScalar       dumr  = sqrt(hr);
  CeedScalar       cl    = sqrt_gravity * duml;
  CeedScalar       cr    = sqrt_gravity * dumr;
  CeedScalar       hhat  = duml * dumr;
  CeedScalar       uhat  = (duml * ul + dumr * ur) / (duml + dumr);
  CeedScalar       vhat  = (duml * vl + dumr * vr) / (duml + dumr);
  CeedScalar       chat  = sqrt(0.5 * gravity * (hl + hr));
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
  FL[1] = ul * uperpl * hl + 0.5 * gravity * hl * hl * cn;
  FL[2] = vl * uperpl * hl + 0.5 * gravity * hl * hl * sn;

  FR[0] = uperpr * hr;
  FR[1] = ur * uperpr * hr + 0.5 * gravity * hr * hr * cn;
  FR[2] = vr * uperpr * hr + 0.5 * gravity * hr * hr * sn;

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
  const SWEContext context            = (SWEContext)ctx;

  const CeedScalar tiny_h  = context->tiny_h;
  const CeedScalar gravity = context->gravity;

  for (CeedInt i = 0; i < Q; i++) {
    SWEState   qL = {q_L[0][i], q_L[1][i], q_L[2][i]};
    SWEState   qR = {q_R[0][i], q_R[1][i], q_R[2][i]};
    CeedScalar flux[3], amax;
    if (qL.h > tiny_h || qR.h > tiny_h) {
      SWERiemannFlux_Roe(gravity, tiny_h, qL, qR, geom[0][i], geom[1][i], flux, &amax);
      for (CeedInt j = 0; j < 3; j++) {
        cell_L[j][i] = flux[j] * geom[2][i];
        cell_R[j][i] = flux[j] * geom[3][i];
      }
    }
  }
  return 0;
}

CEED_QFUNCTION(SWEBoundaryFlux_Reflecting_Roe)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const CeedScalar(*geom)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];  // sn, cn, weight_L
  const CeedScalar(*q_L)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  CeedScalar(*cell_L)[CEED_Q_VLA]     = (CeedScalar(*)[CEED_Q_VLA])out[0];
  const SWEContext context            = (SWEContext)ctx;

  const CeedScalar tiny_h  = context->tiny_h;
  const CeedScalar gravity = context->gravity;

  for (CeedInt i = 0; i < Q; i++) {
    CeedScalar sn = geom[0][i], cn = geom[1][i];
    SWEState   qL   = {q_L[0][i], q_L[1][i], q_L[2][i]};
    if (qL.h > tiny_h) {
      CeedScalar dum1 = sn * sn - cn * cn;
      CeedScalar dum2 = 2.0 * sn * cn;
      SWEState   qR   = {qL.h, qL.hu * dum1 - qL.hv * dum2, -qL.hu * dum2 - qL.hv * dum1};
      CeedScalar flux[3], amax;
      SWERiemannFlux_Roe(gravity, tiny_h, qL, qR, sn, cn, flux, &amax);
      for (CeedInt j = 0; j < 3; j++) {
        cell_L[j][i] = flux[j] * geom[2][i];
      }
    }
  }
  return 0;
}

CEED_QFUNCTION(SWEBoundaryFlux_Outflow_Roe)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  //const CeedScalar gravity            = 9.806;
  const CeedScalar(*geom)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];  // sn, cn, weight_L
  const CeedScalar(*q_L)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  CeedScalar(*cell_L)[CEED_Q_VLA]     = (CeedScalar(*)[CEED_Q_VLA])out[0];
  const SWEContext context            = (SWEContext)ctx;

  const CeedScalar tiny_h  = context->tiny_h;
  const CeedScalar gravity = context->gravity;

  for (CeedInt i = 0; i < Q; i++) {
    CeedScalar sn = geom[0][i], cn = geom[1][i];
    SWEState   qL    = {q_L[0][i], q_L[1][i], q_L[2][i]};
    CeedScalar q     = fabs(qL.hu * cn + qL.hv * sn);
    CeedScalar hR    = pow(q * q / gravity, 1.0 / 3.0);
    CeedScalar speed = sqrt(gravity * hR);
    SWEState   qR    = {hR, hR * speed * cn, hR * speed * sn};
    if (qL.h > tiny_h || qR.h > tiny_h) {
      CeedScalar flux[3], amax;
      SWERiemannFlux_Roe(gravity, tiny_h, qL, qR, sn, cn, flux, &amax);
      for (CeedInt j = 0; j < 3; j++) {
        cell_L[j][i] = flux[j] * geom[2][i];
      }
    }
  }
  return 0;
}

CEED_QFUNCTION(SWESourceTerm)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const CeedScalar(*geom)[CEED_Q_VLA]       = (const CeedScalar(*)[CEED_Q_VLA])in[0];  // dz/dx, dz/dy
  const CeedScalar(*water_src)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[1];  // rain rate
  const CeedScalar(*mannings_n)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];  // mannings coefficient
  const CeedScalar(*riemannf)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[3];  // riemann flux
  const CeedScalar(*q)[CEED_Q_VLA]          = (const CeedScalar(*)[CEED_Q_VLA])in[4];
  CeedScalar(*cell)[CEED_Q_VLA]             = (CeedScalar(*)[CEED_Q_VLA])out[0];
  const SWEContext context                  = (SWEContext)ctx;

  const CeedScalar dt      = context->dtime;
  const CeedScalar tiny_h  = context->tiny_h;
  const CeedScalar gravity = context->gravity;

  for (CeedInt i = 0; i < Q; i++) {
    SWEState         state = {q[0][i], q[1][i], q[2][i]};
    const CeedScalar h     = state.h;
    const CeedScalar hu    = state.hu;
    const CeedScalar hv    = state.hv;

    const CeedScalar u = SafeDiv(state.hu, h, tiny_h);
    const CeedScalar v = SafeDiv(state.hv, h, tiny_h);

    const CeedScalar dz_dx = geom[0][i];
    const CeedScalar dz_dy = geom[1][i];

    const CeedScalar bedx = dz_dx * gravity * h;
    const CeedScalar bedy = dz_dy * gravity * h;

    const CeedScalar Fsum_x = riemannf[1][i];
    const CeedScalar Fsum_y = riemannf[2][i];

    CeedScalar tbx = 0.0, tby = 0.0;
    if (h > tiny_h) {
      const CeedScalar Cd = gravity * Square(mannings_n[0][i]) * pow(h, -1.0 / 3.0);

      const CeedScalar velocity = sqrt(Square(u) + Square(v));

      const CeedScalar tb = Cd * velocity / h;

      const CeedScalar factor = tb / (1.0 + h);

      tbx = (hu + dt * Fsum_x - dt * bedx) * factor;
      tby = (hv + dt * Fsum_y - dt * bedy) * factor;
    }

    cell[0][i] = riemannf[0][i] + water_src[0][i];
    cell[1][i] = riemannf[1][i] - bedx - tbx;
    cell[2][i] = riemannf[2][i] - bedy - tby;
  }
  return 0;
}

#endif  // swe_flux_ceed_h
