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
  CeedInt    sed_ndof;
  CeedInt    flow_ndof;
};

struct SedimentState_ {
  CeedScalar h, hu, hv, hci[MAX_NUM_SEDIMENT_CLASSES];
};
typedef struct SedimentState_ SedimentState;

#ifndef MAX_NUM_SECTION_FIELD_COMPONENTS
#define MAX_NUM_SECTION_FIELD_COMPONENTS 10
#endif

/// computes the flux across an edge using Roe's approximate Riemann solver
/// for flow and sediment transport
CEED_QFUNCTION_HELPER void SedimentRiemannFlux_Roe(const CeedScalar gravity, const CeedScalar tiny_h, SedimentState qL, SedimentState qR,
                                                   CeedScalar sn, CeedScalar cn, CeedInt flow_ndof, CeedInt sed_ndof, CeedScalar flux[],
                                                   CeedScalar *amax) {
  const CeedScalar sqrt_gravity = sqrt(gravity);
  const CeedScalar hl = qL.h, hr = qR.h;

  const CeedScalar ul = SafeDiv(qL.hu, hl, tiny_h), vl = SafeDiv(qL.hv, hl, tiny_h);
  CeedScalar       cil[MAX_NUM_SEDIMENT_CLASSES], cir[MAX_NUM_SEDIMENT_CLASSES];
  for (CeedInt j = 0; j < sed_ndof; ++j) {
    cil[j] = SafeDiv(qL.hci[j], hl, tiny_h);
    cir[j] = SafeDiv(qR.hci[j], hr, tiny_h);
  }
  const CeedScalar ur = SafeDiv(qR.hu, hr, tiny_h), vr = SafeDiv(qR.hv, hr, tiny_h);

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

  for (CeedInt j = 0; j < sed_ndof; j++) {
    cihat[j] = (duml * cil[j] + dumr * cir[j]) / (duml + dumr);
    dch[j]   = cir[j] * hr - cil[j] * hl;
  }

  dW[0] = 0.5 * (dh - hhat * duperp / chat);
  dW[1] = hhat * dupar;
  dW[2] = 0.5 * (dh + hhat * duperp / chat);
  for (CeedInt j = 0; j < sed_ndof; j++) {
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
  for (CeedInt j = 0; j < sed_ndof; j++) {
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
  for (CeedInt j = 0; j < sed_ndof; j++) {
    A[j + 3][j + 3] = a2;
  }

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

  const CeedInt flow_ndof = context->flow_ndof;
  const CeedInt sed_ndof  = context->sed_ndof;
  const CeedInt tot_ndof  = flow_ndof + sed_ndof;

  for (CeedInt i = 0; i < Q; i++) {
    SedimentState qL = {.h = q_L[0][i], .hu = q_L[1][i], .hv = q_L[2][i]};
    SedimentState qR = {.h = q_R[0][i], .hu = q_R[1][i], .hv = q_R[2][i]};
    for (CeedInt j = 0; j < sed_ndof; ++j) {
      qL.hci[j] = q_L[flow_ndof + j][i];
      qR.hci[j] = q_R[3 + j][i];
    }
    CeedScalar flux[tot_ndof], amax;
    if (qL.h > tiny_h || qR.h > tiny_h) {
      SedimentRiemannFlux_Roe(gravity, tiny_h, qL, qR, geom[0][i], geom[1][i], flow_ndof, sed_ndof, flux, &amax);
      for (CeedInt j = 0; j < tot_ndof; j++) {
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

  const CeedInt flow_ndof = context->flow_ndof;
  const CeedInt ѕed_ndof  = context->sed_ndof;
  const CeedInt tot_ndof  = flow_ndof + ѕed_ndof;

  for (CeedInt i = 0; i < Q; i++) {
    SedimentState qL = {.h = q_L[0][i], .hu = q_L[1][i], .hv = q_L[2][i]};
    SedimentState qR = {.h = q_R[0][i], .hu = q_R[1][i], .hv = q_R[2][i]};
    for (CeedInt j = 0; j < ѕed_ndof; ++j) {
      qL.hci[j] = q_L[3 + j][i];
      qR.hci[j] = q_R[3 + j][i];
    }
    if (qL.h > tiny_h) {
      CeedScalar flux[tot_ndof], amax;
      SedimentRiemannFlux_Roe(gravity, tiny_h, qL, qR, geom[0][i], geom[1][i], flow_ndof, ѕed_ndof, flux, &amax);
      for (CeedInt j = 0; j < tot_ndof; j++) {
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

  const CeedInt flow_ndof = context->flow_ndof;
  const CeedInt sed_ndof  = context->sed_ndof;
  const CeedInt tot_ndof  = flow_ndof + sed_ndof;

  for (CeedInt i = 0; i < Q; i++) {
    CeedScalar    sn = geom[0][i], cn = geom[1][i];
    SedimentState qL = {.h = q_L[0][i], .hu = q_L[1][i], .hv = q_L[2][i]};
    for (CeedInt j = 0; j < sed_ndof; ++j) {
      qL.hci[j] = q_L[3 + j][i];
    }
    if (qL.h > tiny_h) {
      CeedScalar    dum1 = sn * sn - cn * cn;
      CeedScalar    dum2 = 2.0 * sn * cn;
      SedimentState qR   = {
            .h  = qL.h,
            .hu = qL.hu * dum1 - qL.hv * dum2,
            .hv = -qL.hu * dum2 - qL.hv * dum1,
      };
      for (CeedInt j = 0; j < sed_ndof; ++j) {
        qR.hci[j] = qL.hci[j];
      }
      CeedScalar flux[tot_ndof], amax;
      SedimentRiemannFlux_Roe(gravity, tiny_h, qL, qR, sn, cn, flow_ndof, sed_ndof, flux, &amax);
      for (CeedInt j = 0; j < tot_ndof; j++) {
        cell_L[j][i] = flux[j] * geom[2][i];
      }
      courant_num[0][i] = -amax * geom[2][i] * dt;
    }
  }
  return 0;
}

// flow and sediment regional source operator Q-function
CEED_QFUNCTION(SedimentSourceTermSemiImplicit)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const CeedScalar(*geom)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[0];  // dz/dx, dz/dy
  const CeedScalar(*ext_src)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[1];  // external source (e.g. rain rate)
  const CeedScalar(*mat_props)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];  // material properties
  const CeedScalar(*riemannf)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[3];  // riemann flux
  const CeedScalar(*q)[CEED_Q_VLA]         = (const CeedScalar(*)[CEED_Q_VLA])in[4];
  CeedScalar(*cell)[CEED_Q_VLA]            = (CeedScalar(*)[CEED_Q_VLA])out[0];
  const SedimentContext context            = (SedimentContext)ctx;

  const CeedScalar dt                      = context->dtime;
  const CeedScalar tiny_h                  = context->tiny_h;
  const CeedScalar gravity                 = context->gravity;
  const CeedScalar kp_constant             = context->kp_constant;
  const CeedScalar settling_velocity       = context->settling_velocity;
  const CeedScalar tau_critical_erosion    = context->tau_critical_erosion;
  const CeedScalar tau_critical_deposition = context->tau_critical_deposition;
  const CeedScalar rhow                    = context->rhow;

  for (CeedInt i = 0; i < Q; i++) {
    SedimentState state = {.h = q[0][i], .hu = q[1][i], .hv = q[2][i]};
    for (CeedInt j = 0; j < context->sed_ndof; ++j) {
      state.hci[j] = q[3 + j][i];
    }
    const CeedScalar h  = state.h;
    const CeedScalar hu = state.hu;
    const CeedScalar hv = state.hv;

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
      const CeedScalar mannings_n = mat_props[OPERATOR_MANNINGS][i];
      const CeedScalar Cd         = gravity * Square(mannings_n) * pow(h, -1.0 / 3.0);

      const CeedScalar velocity = sqrt(Square(u) + Square(v));

      const CeedScalar tb = Cd * velocity / h;

      const CeedScalar factor = tb / (1.0 + dt * tb);

      tbx = (hu + dt * Fsum_x - dt * bedx) * factor;
      tby = (hv + dt * Fsum_y - dt * bedy) * factor;

      for (CeedInt j = 0; j < context->sed_ndof; ++j) {
        const CeedScalar ci    = SafeDiv(state.hci[j], h, tiny_h);
        CeedScalar       tau_b = 0.5 * rhow * Cd * (Square(u) + Square(v));
        CeedScalar       ei    = kp_constant * (tau_b - tau_critical_erosion) / tau_critical_erosion;
        CeedScalar       di    = settling_velocity * ci * (1.0 - tau_b / tau_critical_deposition);
        cell[3 + j][i]         = riemannf[3 + j][i] + (ei - di) + ext_src[3 + j][i];
      }
    }

    cell[0][i] = riemannf[0][i] + ext_src[0][i];
    cell[1][i] = riemannf[1][i] - bedx - tbx + ext_src[1][i];
    cell[2][i] = riemannf[2][i] - bedy - tby + ext_src[2][i];
  }
  return 0;
}

#pragma GCC diagnostic   pop
#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
#pragma clang diagnostic pop

#endif
