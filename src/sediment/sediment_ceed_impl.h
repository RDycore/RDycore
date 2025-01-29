#ifndef SEDIMENT_CEED_IMPL_H
#define SEDIMENT_CEED_IMPL_H

#include <ceed/types.h>

#define Square(x) ((x) * (x))
#define SafeDiv(a, b, tiny) ((b) > (tiny) ? (a) / (b) : 0.0)

// we disable compiler warnings for implicitly-declared math functions known to
// the JIT compiler
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wimplicit-function-declaration"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-function-declaration"

// Q-function context with data attached
typedef struct SedimentContext_ *SedimentContext;
struct SedimentContext_ {
  CeedScalar dtime;
  CeedScalar tiny_h;
  CeedScalar gravity;
  CeedScalar xq2018_threshold;
  CeedScalar kp_constant;
  CeedScalar settling_velocity;
  CeedScalar tau_critical_erosion;
  CeedScalar tau_critical_deposition;
  CeedScalar rhow;
};

struct SedimentState_ {
  CeedScalar h, hu, hv, hci;
};
typedef struct SedimentState_ SedimentState;

#ifndef MAX_NUM_SECTION_FIELD_COMPONENTS
#define MAX_NUM_SECTION_FIELD_COMPONENTS 10
#endif

/// computes the flux across an edge using Roe's approximate Riemann solver
/// for flow and sediment transport
CEED_QFUNCTION_HELPER void SedimentRiemannFlux_Roe(const CeedScalar gravity, const CeedScalar tiny_h, SedimentState qL, SedimentState qR,
                                                   CeedScalar sn, CeedScalar cn, CeedInt sed_ncomp, CeedScalar flux[], CeedScalar *amax) {
  const CeedScalar sqrt_gravity = sqrt(gravity);
  const CeedScalar hl = qL.h, hr = qR.h;

  const CeedScalar ul = SafeDiv(qL.hu, hl, tiny_h), vl = SafeDiv(qL.hv, hl, tiny_h), cil = SafeDiv(qL.hci, hl, tiny_h);
  const CeedScalar ur = SafeDiv(qR.hu, hr, tiny_h), vr = SafeDiv(qR.hv, hr, tiny_h), cir = SafeDiv(qR.hci, hr, tiny_h);

  CeedScalar cihat[MAX_NUM_SECTION_FIELD_COMPONENTS]                               = {0};
  CeedScalar dch[MAX_NUM_SECTION_FIELD_COMPONENTS]                                 = {0};
  CeedScalar dW[MAX_NUM_SECTION_FIELD_COMPONENTS]                                  = {0};
  CeedScalar R[MAX_NUM_SECTION_FIELD_COMPONENTS][MAX_NUM_SECTION_FIELD_COMPONENTS] = {0};
  CeedScalar A[MAX_NUM_SECTION_FIELD_COMPONENTS][MAX_NUM_SECTION_FIELD_COMPONENTS] = {0};
  CeedScalar FL[MAX_NUM_SECTION_FIELD_COMPONENTS]                                  = {0};
  CeedScalar FR[MAX_NUM_SECTION_FIELD_COMPONENTS]                                  = {0};

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

  for (CeedInt j = 0; j < sed_ncomp; j++) {
    cihat[j] = (duml * cil + dumr * cir) / (duml + dumr);  // FIXME: Use cil[j] and cir[j] when sed_ncomp > 1
    dch[j]   = cir * hr - cil * hl;                        // FIXME: Use cil[j] and cir[j] when sed_ncomp > 1
  }

  dW[0] = 0.5 * (dh - hhat * duperp / chat);
  dW[1] = hhat * dupar;
  dW[2] = 0.5 * (dh + hhat * duperp / chat);
  for (CeedInt j = 0; j < sed_ncomp; j++) {
    dW[j + 3] = dch[j] - cihat[j] * dh;
  }

  CeedScalar uperpl = ul * cn + vl * sn;
  CeedScalar uperpr = ur * cn + vr * sn;
  CeedScalar al1    = uperpl - cl;
  CeedScalar al3    = uperpl + cl;
  CeedScalar ar1    = uperpr - cr;
  CeedScalar ar3    = uperpr + cr;

  R[0][0] = 1.0;
  R[0][1] = 0.0;
  R[0][2] = 1.0;
  R[1][0] = uhat - chat * cn;
  R[1][1] = -sn;
  R[1][2] = uhat + chat * cn;
  R[2][0] = vhat - chat * sn;
  R[2][1] = cn;
  R[2][2] = vhat + chat * sn;
  for (CeedInt j = 0; j < sed_ncomp; j++) {
    R[j + 3][0]     = cihat[j];
    R[j + 3][2]     = cihat[j];
    R[j + 3][j + 3] = 1.0;
  }

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
  A[0][0] = a1;
  A[1][1] = a2;
  A[2][2] = a3;
  for (CeedInt j = 0; j < sed_ncomp; j++) {
    A[j + 3][j + 3] = a2;
  }

  FL[0] = uperpl * hl;
  FL[1] = ul * uperpl * hl + 0.5 * gravity * hl * hl * cn;
  FL[2] = vl * uperpl * hl + 0.5 * gravity * hl * hl * sn;

  FR[0] = uperpr * hr;
  FR[1] = ur * uperpr * hr + 0.5 * gravity * hr * hr * cn;
  FR[2] = vr * uperpr * hr + 0.5 * gravity * hr * hr * sn;

  for (CeedInt j = 0; j < sed_ncomp; j++) {
    FL[j + 3] = hl * uperpl * cil;  // FIXME: Use cil[j] when sed_ncomp > 1
    FR[j + 3] = hr * uperpr * cir;  // FIXME: Use cir[j] when sed_ncomp > 1
  }

  // flux = 0.5*(FL + FR - matmul(R,matmul(A,dW))
  CeedInt soln_ncomp = 3 + sed_ncomp;
  for (CeedInt dof1 = 0; dof1 < soln_ncomp; dof1++) {
    flux[dof1] = 0.5 * (FL[dof1] + FR[dof1]);

    for (CeedInt dof2 = 0; dof2 < soln_ncomp; dof2++) {
      flux[dof1] = flux[dof1] - 0.5 * R[dof1][dof2] * A[dof2][dof2] * dW[dof2];
    }
  }

  *amax = chat + fabs(uperp);
}

// The following Q functions use C99 VLA features for shaping multidimensional
// arrays, which don't have the same drawbacks as VLA allocations.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla"

// flow and sediment flux operator Q-function for interior edges
CEED_QFUNCTION(SedimentFlux_Roe)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const CeedScalar(*geom)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[0];  // sn, cn, weight_L, weight_R
  const CeedScalar(*q_L)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*q_R)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  CeedScalar(*cell_L)[CEED_Q_VLA]      = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*cell_R)[CEED_Q_VLA]      = (CeedScalar(*)[CEED_Q_VLA])out[1];
  CeedScalar(*accum_flux)[CEED_Q_VLA]  = (CeedScalar(*)[CEED_Q_VLA])out[2];
  CeedScalar(*courant_num)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[3];
  const SedimentContext context        = (SedimentContext)ctx;

  const CeedScalar dt      = context->dtime;
  const CeedScalar tiny_h  = context->tiny_h;
  const CeedScalar gravity = context->gravity;

  const CeedInt ndof_flow     = 3;
  const CeedInt ndof_sediment = 1;  // FIXME: Need to be part of 'context'
  const CeedInt ndof_total    = ndof_flow + ndof_sediment;

  for (CeedInt i = 0; i < Q; i++) {
    SedimentState qL = {q_L[0][i], q_L[1][i], q_L[2][i], q_L[3][i]};
    SedimentState qR = {q_R[0][i], q_R[1][i], q_R[2][i], q_R[3][i]};
    CeedScalar    flux[ndof_total], amax;
    if (qL.h > tiny_h || qR.h > tiny_h) {
      SedimentRiemannFlux_Roe(gravity, tiny_h, qL, qR, geom[0][i], geom[1][i], ndof_sediment, flux, &amax);
      for (CeedInt j = 0; j < ndof_total; j++) {
        cell_L[j][i]     = flux[j] * geom[2][i];
        cell_R[j][i]     = flux[j] * geom[3][i];
        accum_flux[j][i] = flux[j];
      }
      courant_num[0][i] = -amax * geom[2][i] * dt;
      courant_num[1][i] = amax * geom[3][i] * dt;
    }
  }
  return 0;
}

// flow and sediment flux operator Q-function for boundary edges on which dirichlet condition is applied
CEED_QFUNCTION(SedimentBoundaryFlux_Dirichlet_Roe)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const CeedScalar(*geom)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[0];  // sn, cn, weight_L
  const CeedScalar(*q_L)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*q_R)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[2];  // Dirichlet boundary values
  CeedScalar(*cell_L)[CEED_Q_VLA]      = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*accum_flux)[CEED_Q_VLA]  = (CeedScalar(*)[CEED_Q_VLA])out[1];
  CeedScalar(*courant_num)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[2];
  const SedimentContext context        = (SedimentContext)ctx;

  const CeedScalar dt      = context->dtime;
  const CeedScalar tiny_h  = context->tiny_h;
  const CeedScalar gravity = context->gravity;

  const CeedInt ndof_flow     = 3;
  const CeedInt ndof_sediment = 1;  // FIXME: Need to be part of 'context'
  const CeedInt ndof_total    = ndof_flow + ndof_sediment;

  for (CeedInt i = 0; i < Q; i++) {
    SedimentState qL = {q_L[0][i], q_L[1][i], q_L[2][i], q_L[3][i]};
    SedimentState qR = {q_R[0][i], q_R[1][i], q_R[2][i], q_R[3][i]};
    if (qL.h > tiny_h) {
      CeedScalar flux[ndof_total], amax;
      SedimentRiemannFlux_Roe(gravity, tiny_h, qL, qR, geom[0][i], geom[1][i], ndof_sediment, flux, &amax);
      for (CeedInt j = 0; j < ndof_total; j++) {
        cell_L[j][i]     = flux[j] * geom[2][i];
        accum_flux[j][i] = flux[j];
      }
      courant_num[0][i] = -amax * geom[2][i] * dt;
    }
  }
  return 0;
}

// flow and sediment flux operator Q-function for boundary edges on which reflecting wall condition is applied
CEED_QFUNCTION(SedimentBoundaryFlux_Reflecting_Roe)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const CeedScalar(*geom)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[0];  // sn, cn, weight_L
  const CeedScalar(*q_L)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  CeedScalar(*cell_L)[CEED_Q_VLA]      = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*courant_num)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[2];
  const SedimentContext context        = (SedimentContext)ctx;

  const CeedScalar dt      = context->dtime;
  const CeedScalar tiny_h  = context->tiny_h;
  const CeedScalar gravity = context->gravity;

  const CeedInt ndof_flow     = 3;
  const CeedInt ndof_sediment = 1;  // FIXME: Need to be part of 'context'
  const CeedInt ndof_total    = ndof_flow + ndof_sediment;

  for (CeedInt i = 0; i < Q; i++) {
    CeedScalar    sn = geom[0][i], cn = geom[1][i];
    SedimentState qL = {q_L[0][i], q_L[1][i], q_L[2][i], q_L[3][i]};
    if (qL.h > tiny_h) {
      CeedScalar    dum1 = sn * sn - cn * cn;
      CeedScalar    dum2 = 2.0 * sn * cn;
      SedimentState qR   = {qL.h, qL.hu * dum1 - qL.hv * dum2, -qL.hu * dum2 - qL.hv * dum1, qL.hci};
      CeedScalar    flux[ndof_total], amax;
      SedimentRiemannFlux_Roe(gravity, tiny_h, qL, qR, sn, cn, ndof_sediment, flux, &amax);
      for (CeedInt j = 0; j < ndof_total; j++) {
        cell_L[j][i] = flux[j] * geom[2][i];
      }
      courant_num[0][i] = -amax * geom[2][i] * dt;
    }
  }
  return 0;
}

// flow and sediment regional source operator Q-function
CEED_QFUNCTION(SedimentSourceTermSemiImplicit)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const CeedScalar(*geom)[CEED_Q_VLA]       = (const CeedScalar(*)[CEED_Q_VLA])in[0];  // dz/dx, dz/dy
  const CeedScalar(*swe_src)[CEED_Q_VLA]    = (const CeedScalar(*)[CEED_Q_VLA])in[1];  // external source (e.g. rain rate)
  const CeedScalar(*mannings_n)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];  // mannings coefficient
  const CeedScalar(*riemannf)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[3];  // riemann flux
  const CeedScalar(*q)[CEED_Q_VLA]          = (const CeedScalar(*)[CEED_Q_VLA])in[4];
  CeedScalar(*cell)[CEED_Q_VLA]             = (CeedScalar(*)[CEED_Q_VLA])out[0];
  const SedimentContext context             = (SedimentContext)ctx;

  const CeedScalar dt                      = context->dtime;
  const CeedScalar tiny_h                  = context->tiny_h;
  const CeedScalar gravity                 = context->gravity;
  const CeedScalar kp_constant             = context->kp_constant;
  const CeedScalar settling_velocity       = context->settling_velocity;
  const CeedScalar tau_critical_erosion    = context->tau_critical_erosion;
  const CeedScalar tau_critical_deposition = context->tau_critical_deposition;
  const CeedScalar rhow                    = context->rhow;

  for (CeedInt i = 0; i < Q; i++) {
    SedimentState    state = {q[0][i], q[1][i], q[2][i], q[3][i]};
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

      const CeedScalar factor = tb / (1.0 + dt * tb);

      tbx = (hu + dt * Fsum_x - dt * bedx) * factor;
      tby = (hv + dt * Fsum_y - dt * bedy) * factor;

      const CeedScalar ci    = SafeDiv(state.hci, h, tiny_h);
      CeedScalar       tau_b = 0.5 * rhow * Cd * (Square(u) + Square(v));
      CeedScalar       ei    = kp_constant * (tau_b - tau_critical_erosion) / tau_critical_erosion;
      CeedScalar       di    = settling_velocity * ci * (1.0 - tau_b / tau_critical_deposition);
      cell[3][i]             = riemannf[3][i] + (ei - di) + swe_src[3][i];
    }

    cell[0][i] = riemannf[0][i] + swe_src[0][i];
    cell[1][i] = riemannf[1][i] - bedx - tbx + swe_src[1][i];
    cell[2][i] = riemannf[2][i] - bedy - tby + swe_src[2][i];
  }
  return 0;
}

#pragma GCC diagnostic   pop
#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
#pragma clang diagnostic pop

#endif
