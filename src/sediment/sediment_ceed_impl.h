#ifndef SEDIMENT_CEED_IMPL_H
#define SEDIMENT_CEED_IMPL_H

#include "sediment_ceed.h"

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

#include "sediment_roe_ceed_impl.h"

// The following Q functions use C99 VLA features for shaping multidimensional
// arrays, which don't have the same drawbacks as VLA allocations.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla"

// flow and sediment flux operator Q-function for interior edges
CEED_QFUNCTION_HELPER int SedimentFlux(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[], RiemannFluxType flux_type) {
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
    SedimentState qL = {q_L[0][i], q_L[1][i], q_L[2][i]};
    SedimentState qR = {q_R[0][i], q_R[1][i], q_R[2][i]};
    for (CeedInt j = 0; j < sed_ndof; ++j) {
      qL.hci[j] = q_L[flow_ndof + j][i];
      qR.hci[j] = q_R[flow_ndof + j][i];
    }
    CeedScalar flux[MAX_NUM_FIELD_COMPONENTS], amax;
    if (qL.h > tiny_h || qR.h > tiny_h) {
      switch (flux_type) {
        case RIEMANN_FLUX_ROE:
          SedimentRiemannFlux_Roe(gravity, tiny_h, qL, qR, geom[0][i], geom[1][i], flow_ndof, sed_ndof, flux, &amax);
          break;
      }
      for (CeedInt j = 0; j < tot_ndof; j++) {
        cell_L[j][i]     = flux[j] * geom[2][i];
        cell_R[j][i]     = flux[j] * geom[3][i];
        accum_flux[j][i] = flux[j];
      }
      courant_num[0][i] = -amax * geom[2][i] * dt;
      courant_num[1][i] = amax * geom[3][i] * dt;
    } else {
      for (CeedInt j = 0; j < tot_ndof; j++) {
        cell_L[j][i]     = 0.0;
        cell_R[j][i]     = 0.0;
        accum_flux[j][i] = 0.0;
      }
      courant_num[0][i] = 0.0;
      courant_num[1][i] = 0.0;
    }
  }
  return 0;
}

CEED_QFUNCTION(SedimentFlux_Roe)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  return SedimentFlux(ctx, Q, in, out, RIEMANN_FLUX_ROE);
}

// flow and sediment flux operator Q-function for boundary edges on which dirichlet condition is applied
CEED_QFUNCTION_HELPER int SedimentBoundaryFlux_Dirichlet(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[],
                                                         RiemannFluxType flux_type) {
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
  const CeedInt sed_ndof  = context->sed_ndof;
  const CeedInt tot_ndof  = flow_ndof + sed_ndof;

  for (CeedInt i = 0; i < Q; i++) {
    SedimentState qL = {q_L[0][i], q_L[1][i], q_L[2][i]};
    SedimentState qR = {q_R[0][i], q_R[1][i], q_R[2][i]};
    for (CeedInt j = 0; j < sed_ndof; ++j) {
      qL.hci[j] = q_L[flow_ndof + j][i];
      qR.hci[j] = q_R[flow_ndof + j][i];
    }
    if (qL.h > tiny_h) {
      CeedScalar flux[MAX_NUM_FIELD_COMPONENTS], amax;
      switch (flux_type) {
        case RIEMANN_FLUX_ROE:
          SedimentRiemannFlux_Roe(gravity, tiny_h, qL, qR, geom[0][i], geom[1][i], flow_ndof, sed_ndof, flux, &amax);
          break;
      }
      for (CeedInt j = 0; j < tot_ndof; j++) {
        cell_L[j][i]     = flux[j] * geom[2][i];
        accum_flux[j][i] = flux[j];
      }
      courant_num[0][i] = -amax * geom[2][i] * dt;
    } else {
      for (CeedInt j = 0; j < tot_ndof; j++) {
        cell_L[j][i]     = 0.0;
        accum_flux[j][i] = 0.0;
      }
      courant_num[0][i] = 0.0;
      courant_num[1][i] = 0.0;
    }
  }
  return 0;
}

CEED_QFUNCTION(SedimentBoundaryFlux_Dirichlet_Roe)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  return SedimentBoundaryFlux_Dirichlet(ctx, Q, in, out, RIEMANN_FLUX_ROE);
}

// flow and sediment flux operator Q-function for boundary edges on which reflecting wall condition is applied
CEED_QFUNCTION_HELPER int SedimentBoundaryFlux_Reflecting(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[],
                                                          RiemannFluxType flux_type) {
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
    SedimentState qL = {q_L[0][i], q_L[1][i], q_L[2][i]};
    for (CeedInt j = 0; j < sed_ndof; ++j) {
      qL.hci[j] = q_L[flow_ndof + j][i];
    }
    if (qL.h > tiny_h) {
      CeedScalar    dum1 = sn * sn - cn * cn;
      CeedScalar    dum2 = 2.0 * sn * cn;
      SedimentState qR   = {qL.h, qL.hu * dum1 - qL.hv * dum2, -qL.hu * dum2 - qL.hv * dum1};
      for (CeedInt j = 0; j < sed_ndof; ++j) {
        qR.hci[j] = qL.hci[j];
      }
      CeedScalar flux[MAX_NUM_FIELD_COMPONENTS], amax;
      switch (flux_type) {
        case RIEMANN_FLUX_ROE:
          SedimentRiemannFlux_Roe(gravity, tiny_h, qL, qR, sn, cn, flow_ndof, sed_ndof, flux, &amax);
          break;
      }
      for (CeedInt j = 0; j < tot_ndof; j++) {
        cell_L[j][i] = flux[j] * geom[2][i];
      }
      courant_num[0][i] = -amax * geom[2][i] * dt;
    } else {
      for (CeedInt j = 0; j < tot_ndof; j++) {
        cell_L[j][i] = 0.0;
      }
      courant_num[0][i] = 0.0;
    }
  }
  return 0;
}

CEED_QFUNCTION(SedimentBoundaryFlux_Reflecting_Roe)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  return SedimentBoundaryFlux_Reflecting(ctx, Q, in, out, RIEMANN_FLUX_ROE);
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

  const CeedInt flow_ndof = context->flow_ndof;

  const CeedScalar dt                      = context->dtime;
  const CeedScalar tiny_h                  = context->tiny_h;
  const CeedScalar gravity                 = context->gravity;
  const CeedScalar kp_constant             = context->kp_constant;
  const CeedScalar settling_velocity       = context->settling_velocity;
  const CeedScalar tau_critical_erosion    = context->tau_critical_erosion;
  const CeedScalar tau_critical_deposition = context->tau_critical_deposition;
  const CeedScalar rhow                    = context->rhow;

  for (CeedInt i = 0; i < Q; i++) {
    SedimentState state = {q[0][i], q[1][i], q[2][i]};
    for (CeedInt j = 0; j < context->sed_ndof; ++j) {
      state.hci[j] = q[flow_ndof + j][i];
    }
    const CeedScalar h  = state.h;
    const CeedScalar hu = state.hu;
    const CeedScalar hv = state.hv;

    const CeedScalar u = SafeDiv(state.hu, h, h, tiny_h);
    const CeedScalar v = SafeDiv(state.hv, h, h, tiny_h);

    const CeedScalar dz_dx = geom[0][i];
    const CeedScalar dz_dy = geom[1][i];

    const CeedScalar bedx = dz_dx * gravity * h;
    const CeedScalar bedy = dz_dy * gravity * h;

    const CeedScalar Fsum_x = riemannf[1][i];
    const CeedScalar Fsum_y = riemannf[2][i];

    CeedScalar tbx = 0.0, tby = 0.0;
    if (h > tiny_h) {
      const CeedScalar mannings_n = mat_props[MATERIAL_PROPERTY_MANNINGS][i];
      const CeedScalar Cd         = gravity * Square(mannings_n) * pow(h, -1.0 / 3.0);

      const CeedScalar velocity = sqrt(Square(u) + Square(v));

      const CeedScalar tb = Cd * velocity / h;

      const CeedScalar factor = tb / (1.0 + dt * tb);

      tbx = (hu + dt * Fsum_x - dt * bedx) * factor;
      tby = (hv + dt * Fsum_y - dt * bedy) * factor;

      for (CeedInt j = 0; j < context->sed_ndof; ++j) {
        const CeedScalar ci    = SafeDiv(state.hci[j], h, h, tiny_h);
        CeedScalar       tau_b = 0.5 * rhow * Cd * (Square(u) + Square(v));
        CeedScalar       ei    = kp_constant * (tau_b - tau_critical_erosion) / tau_critical_erosion;
        CeedScalar       di    = settling_velocity * ci * (1.0 - tau_b / tau_critical_deposition);
        cell[flow_ndof + j][i] = riemannf[flow_ndof + j][i] + (ei - di) + ext_src[flow_ndof + j][i];
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
