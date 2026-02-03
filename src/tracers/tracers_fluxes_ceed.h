#ifndef TRACERS_FLUXES_CEED_H
#define TRACERS_FLUXES_CEED_H

#include "tracers_types_ceed.h"

// we disable compiler warnings for implicitly-declared math functions known to
// the JIT compiler
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wimplicit-function-declaration"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-function-declaration"

#include "tracers_roe_flux_ceed.h"

// The following Q functions use C99 VLA features for shaping multidimensional
// arrays, which don't have the same drawbacks as VLA allocations.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla"

// flow and tracers flux operator Q-function for interior edges
CEED_QFUNCTION_HELPER int TracersFlux(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[], RiemannFluxType flux_type) {
  const CeedScalar(*geom)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[0];  // sn, cn, weight_L, weight_R
  const CeedScalar(*q_L)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*q_R)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  CeedScalar(*cell_L)[CEED_Q_VLA]      = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*cell_R)[CEED_Q_VLA]      = (CeedScalar(*)[CEED_Q_VLA])out[1];
  CeedScalar(*accum_flux)[CEED_Q_VLA]  = (CeedScalar(*)[CEED_Q_VLA])out[2];
  CeedScalar(*courant_num)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[3];
  const TracersContext context        = (TracersContext)ctx;

  const CeedScalar dt      = context->dtime;
  const CeedScalar tiny_h  = context->tiny_h;
  const CeedScalar gravity = context->gravity;

  const CeedInt flow_ndof = context->flow_ndof;
  const CeedInt tracers_ndof  = context->tracers_ndof;
  const CeedInt tot_ndof  = flow_ndof + tracers_ndof;

  for (CeedInt i = 0; i < Q; i++) {
    TracersState qL = {q_L[0][i], q_L[1][i], q_L[2][i]};
    TracersState qR = {q_R[0][i], q_R[1][i], q_R[2][i]};
    for (CeedInt j = 0; j < tracers_ndof; ++j) {
      qL.hci[j] = q_L[flow_ndof + j][i];
      qR.hci[j] = q_R[flow_ndof + j][i];
    }
    CeedScalar flux[MAX_NUM_FIELD_COMPONENTS], amax;
    if (qL.h > tiny_h || qR.h > tiny_h) {
      switch (flux_type) {
        case RIEMANN_FLUX_ROE:
          TracersRiemannFlux_Roe(gravity, tiny_h, qL, qR, geom[0][i], geom[1][i], flow_ndof, tracers_ndof, flux, &amax);
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

CEED_QFUNCTION(TracersFlux_Roe)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  return TracersFlux(ctx, Q, in, out, RIEMANN_FLUX_ROE);
}

// flow and tracers flux operator Q-function for boundary edges on which dirichlet condition is applied
CEED_QFUNCTION_HELPER int TracersBoundaryFlux_Dirichlet(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[],
                                                         RiemannFluxType flux_type) {
  const CeedScalar(*geom)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[0];  // sn, cn, weight_L
  const CeedScalar(*q_L)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*q_R)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[2];  // Dirichlet boundary values
  CeedScalar(*cell_L)[CEED_Q_VLA]      = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*accum_flux)[CEED_Q_VLA]  = (CeedScalar(*)[CEED_Q_VLA])out[1];
  CeedScalar(*courant_num)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[2];
  const TracersContext context        = (TracersContext)ctx;

  const CeedScalar dt      = context->dtime;
  const CeedScalar tiny_h  = context->tiny_h;
  const CeedScalar gravity = context->gravity;

  const CeedInt flow_ndof = context->flow_ndof;
  const CeedInt tracers_ndof  = context->tracers_ndof;
  const CeedInt tot_ndof  = flow_ndof + tracers_ndof;

  for (CeedInt i = 0; i < Q; i++) {
    TracersState qL = {q_L[0][i], q_L[1][i], q_L[2][i]};
    TracersState qR = {q_R[0][i], q_R[1][i], q_R[2][i]};
    for (CeedInt j = 0; j < tracers_ndof; ++j) {
      qL.hci[j] = q_L[flow_ndof + j][i];
      qR.hci[j] = q_R[flow_ndof + j][i];
    }
    if (qL.h > tiny_h || qR.h > tiny_h) {
      CeedScalar flux[MAX_NUM_FIELD_COMPONENTS], amax;
      switch (flux_type) {
        case RIEMANN_FLUX_ROE:
          TracersRiemannFlux_Roe(gravity, tiny_h, qL, qR, geom[0][i], geom[1][i], flow_ndof, tracers_ndof, flux, &amax);
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

CEED_QFUNCTION(TracersBoundaryFlux_Dirichlet_Roe)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  return TracersBoundaryFlux_Dirichlet(ctx, Q, in, out, RIEMANN_FLUX_ROE);
}

// flow and tracers flux operator Q-function for boundary edges on which reflecting wall condition is applied
CEED_QFUNCTION_HELPER int TracersBoundaryFlux_Reflecting(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[],
                                                          RiemannFluxType flux_type) {
  const CeedScalar(*geom)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[0];  // sn, cn, weight_L
  const CeedScalar(*q_L)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  CeedScalar(*cell_L)[CEED_Q_VLA]      = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*courant_num)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[2];
  const TracersContext context        = (TracersContext)ctx;

  const CeedScalar dt      = context->dtime;
  const CeedScalar tiny_h  = context->tiny_h;
  const CeedScalar gravity = context->gravity;

  const CeedInt flow_ndof = context->flow_ndof;
  const CeedInt tracers_ndof  = context->tracers_ndof;
  const CeedInt tot_ndof  = flow_ndof + tracers_ndof;

  for (CeedInt i = 0; i < Q; i++) {
    CeedScalar    sn = geom[0][i], cn = geom[1][i];
    TracersState qL = {q_L[0][i], q_L[1][i], q_L[2][i]};
    for (CeedInt j = 0; j < tracers_ndof; ++j) {
      qL.hci[j] = q_L[flow_ndof + j][i];
    }
    if (qL.h > tiny_h) {
      CeedScalar    dum1 = sn * sn - cn * cn;
      CeedScalar    dum2 = 2.0 * sn * cn;
      TracersState qR   = {qL.h, qL.hu * dum1 - qL.hv * dum2, -qL.hu * dum2 - qL.hv * dum1};
      for (CeedInt j = 0; j < tracers_ndof; ++j) {
        qR.hci[j] = qL.hci[j];
      }
      CeedScalar flux[MAX_NUM_FIELD_COMPONENTS], amax;
      switch (flux_type) {
        case RIEMANN_FLUX_ROE:
          TracersRiemannFlux_Roe(gravity, tiny_h, qL, qR, sn, cn, flow_ndof, tracers_ndof, flux, &amax);
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

CEED_QFUNCTION(TracersBoundaryFlux_Reflecting_Roe)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  return TracersBoundaryFlux_Reflecting(ctx, Q, in, out, RIEMANN_FLUX_ROE);
}

#pragma GCC diagnostic   pop
#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
#pragma clang diagnostic pop

#endif
