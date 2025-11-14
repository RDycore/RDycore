#ifndef SEDIMENT_SOURCES_CEED_H
#define SEDIMENT_SOURCES_CEED_H

#include "sediment_types_ceed.h"

// we disable compiler warnings for implicitly-declared math functions known to
// the JIT compiler
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wimplicit-function-declaration"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-function-declaration"

// The following Q functions use C99 VLA features for shaping multidimensional
// arrays, which don't have the same drawbacks as VLA allocations.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla"

CEED_QFUNCTION(SedimentFluxDivergenceSourceTerm)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const CeedScalar(*riemannf)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[3];  // riemann flux
  const CeedScalar(*q)[CEED_Q_VLA]         = (const CeedScalar(*)[CEED_Q_VLA])in[4];
  CeedScalar(*cell)[CEED_Q_VLA]            = (CeedScalar(*)[CEED_Q_VLA])out[0];
  const SedimentContext context            = (SedimentContext)ctx;

  const CeedInt flow_ndof = context->flow_ndof;

  const CeedScalar tiny_h                  = context->tiny_h;

  for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar h  = q[0][i];

    if (h > tiny_h) {
      for (CeedInt j = 0; j < context->sed_ndof; ++j) {
        cell[flow_ndof + j][i] += riemannf[flow_ndof + j][i];
      }
    }

    cell[0][i] += riemannf[0][i];
    cell[1][i] += riemannf[1][i];
    cell[2][i] += riemannf[2][i];
  }
  return 0;
}

CEED_QFUNCTION(SedimentExternalSourceTerm)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const CeedScalar(*geom)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[0];  // dz/dx, dz/dy
  const CeedScalar(*ext_src)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[1];  // external source (e.g. rain rate)
  const CeedScalar(*mat_props)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];  // material properties
  const CeedScalar(*riemannf)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[3];  // riemann flux
  const CeedScalar(*q)[CEED_Q_VLA]         = (const CeedScalar(*)[CEED_Q_VLA])in[4];
  CeedScalar(*cell)[CEED_Q_VLA]            = (CeedScalar(*)[CEED_Q_VLA])out[0];
  const SedimentContext context            = (SedimentContext)ctx;

  const CeedInt flow_ndof = context->flow_ndof;
  const CeedScalar tiny_h = context->tiny_h;

  for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar h  = q[0][i];

    if (h > tiny_h) {
      for (CeedInt j = 0; j < context->sed_ndof; ++j) {
        cell[flow_ndof + j][i] += ext_src[flow_ndof + j][i];
      }
    }

    cell[0][i] += ext_src[0][i];
    cell[1][i] += ext_src[1][i];
    cell[2][i] += ext_src[2][i];
  }
  return 0;
}

CEED_QFUNCTION(SWEBedElevationSlopeSourceTerm)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const CeedScalar(*geom)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[0];  // dz/dx, dz/dy
  const CeedScalar(*q)[CEED_Q_VLA]         = (const CeedScalar(*)[CEED_Q_VLA])in[4];
  CeedScalar(*cell)[CEED_Q_VLA]            = (CeedScalar(*)[CEED_Q_VLA])out[0];
  const SedimentContext context            = (SedimentContext)ctx;

  const CeedScalar gravity                 = context->gravity;

  for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar h  = q[0][i];

    const CeedScalar dz_dx = geom[0][i];
    const CeedScalar dz_dy = geom[1][i];

    const CeedScalar bedx = dz_dx * gravity * h;
    const CeedScalar bedy = dz_dy * gravity * h;

    cell[1][i] -= bedx;
    cell[2][i] -= bedy;
  }
  return 0;
}

CEED_QFUNCTION(SedimentBedFrictionRoughnessSourceTermSemiImplicit)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const CeedScalar(*mat_props)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];  // material properties
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

    CeedScalar tbx = 0.0, tby = 0.0;
    if (h > tiny_h) {
      const CeedScalar mannings_n = mat_props[MATERIAL_PROPERTY_MANNINGS][i];
      const CeedScalar Cd         = gravity * Square(mannings_n) * pow(h, -1.0 / 3.0);

      const CeedScalar velocity = sqrt(Square(u) + Square(v));

      const CeedScalar tb = Cd * velocity / h;

      const CeedScalar factor = tb / (1.0 + dt * tb);

      // gather other source contributions, since we're the last term
      const CeedScalar dsxdt = cell[1][i];
      const CeedScalar dsydt = cell[2][i];

      tbx = (hu + dt * dsxdt) * factor;
      tby = (hv + dt * dsydt) * factor;

      for (CeedInt j = 0; j < context->sed_ndof; ++j) {
        const CeedScalar ci    = SafeDiv(state.hci[j], h, h, tiny_h);
        CeedScalar       tau_b = 0.5 * rhow * Cd * (Square(u) + Square(v));
        CeedScalar       ei    = kp_constant * (tau_b - tau_critical_erosion) / tau_critical_erosion;
        CeedScalar       di    = settling_velocity * ci * (1.0 - tau_b / tau_critical_deposition);
        cell[flow_ndof + j][i] += (ei - di); // FIXME: is this where this term belongs?
      }
    }

    cell[1][i] -= tbx;
    cell[2][i] -= tby;
  }
  return 0;
}

#pragma GCC diagnostic   pop
#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
#pragma clang diagnostic pop

#endif
