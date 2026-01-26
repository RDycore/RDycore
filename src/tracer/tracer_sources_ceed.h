#ifndef TRACER_SOURCES_CEED_H
#define TRACER_SOURCES_CEED_H

#include "private/config.h"
#include "tracer_types_ceed.h"

// supported bed friction source term methods
typedef enum {
  TRACER_BED_FRICTION_NONE,
  TRACER_BED_FRICTION_SEMI_IMPLICIT,
  TRACER_BED_FRICTION_IMPLICIT_XQ2018,
} TracerBedFrictionType;

#ifndef Square
#define Square(x) ((x) * (x))
#endif
#ifndef SafeDiv
#define SafeDiv(a, b, c, tiny) ((c) > (tiny) ? (a) / (b) : 0.0)
#endif

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

int CEED_QFUNCTION_HELPER TracerSemiImplicitBedFrictionRoughness(TracerContext context, TracerState state, CeedScalar mannings_n, CeedScalar Fsum_x,
                                                                 CeedScalar Fsum_y, CeedScalar bedx, CeedScalar bedy, CeedScalar *tbx,
                                                                 CeedScalar *tby, CeedScalar e[MAX_NUM_SEDIMENT_CLASSES],
                                                                 CeedScalar d[MAX_NUM_SEDIMENT_CLASSES]) {
  const CeedScalar dt                      = context->dtime;
  const CeedScalar tiny_h                  = context->tiny_h;
  const CeedScalar gravity                 = context->gravity;
  const CeedScalar kp_constant             = context->kp_constant;
  const CeedScalar settling_velocity       = context->settling_velocity;
  const CeedScalar tau_critical_erosion    = context->tau_critical_erosion;
  const CeedScalar tau_critical_deposition = context->tau_critical_deposition;
  const CeedScalar rhow                    = context->rhow;

  const CeedScalar h  = state.h;
  const CeedScalar hu = state.hu;
  const CeedScalar hv = state.hv;

  const CeedScalar u = SafeDiv(state.hu, h, h, tiny_h);
  const CeedScalar v = SafeDiv(state.hv, h, h, tiny_h);

  const CeedScalar Cd       = gravity * Square(mannings_n) * pow(h, -1.0 / 3.0);
  const CeedScalar velocity = sqrt(Square(u) + Square(v));

  const CeedScalar tb = Cd * velocity / h;

  const CeedScalar factor = tb / (1.0 + dt * tb);

  *tbx = (hu + dt * (Fsum_x - bedx)) * factor;
  *tby = (hv + dt * (Fsum_y - bedy)) * factor;

  for (CeedInt j = 0; j < context->tracer_ndof; ++j) {
    const CeedScalar ci    = SafeDiv(state.hci[j], h, h, tiny_h);
    CeedScalar       tau_b = 0.5 * rhow * Cd * (Square(u) + Square(v));
    e[j]                   = kp_constant * (tau_b - tau_critical_erosion) / tau_critical_erosion;
    d[j]                   = settling_velocity * ci * (1.0 - tau_b / tau_critical_deposition);
  }

  return 0;
}

CEED_QFUNCTION_HELPER int TracerSources(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[],
                                        TracerBedFrictionType bed_friction_type) {
  // inputs
  const CeedScalar(*geom)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[0];  // dz/dx, dz/dy
  const CeedScalar(*ext_src)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[1];  // external source (e.g. rain rate)
  const CeedScalar(*mat_props)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];  // material properties
  const CeedScalar(*riemannf)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[3];  // riemann flux
  const CeedScalar(*q)[CEED_Q_VLA]         = (const CeedScalar(*)[CEED_Q_VLA])in[4];

  // outputs
  CeedScalar(*sources)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const TracerContext context   = (TracerContext)ctx;
  const CeedInt       flow_ndof = context->flow_ndof;
  const CeedScalar    tiny_h    = context->tiny_h;
  const CeedScalar    gravity   = context->gravity;

  for (CeedInt i = 0; i < Q; i++) {
    TracerState state = {q[0][i], q[1][i], q[2][i]};
    for (CeedInt j = 0; j < context->tracer_ndof; ++j) {
      state.hci[j] = q[flow_ndof + j][i];
    }

    const CeedScalar dz_dx = geom[0][i];
    const CeedScalar dz_dy = geom[1][i];

    const CeedScalar bedx = dz_dx * gravity * state.h;
    const CeedScalar bedy = dz_dy * gravity * state.h;

    // bed friction roughness
    CeedScalar tbx = 0.0, tby = 0.0;

    // detachment/deposition rates
    CeedScalar e[MAX_NUM_SEDIMENT_CLASSES] = {0}, d[MAX_NUM_SEDIMENT_CLASSES] = {0};

    if (state.h > tiny_h) {
      const CeedScalar Fsum_x     = riemannf[1][i];
      const CeedScalar Fsum_y     = riemannf[2][i];
      const CeedScalar mannings_n = mat_props[MATERIAL_PROPERTY_MANNINGS][i];

      switch (bed_friction_type) {
        case TRACER_BED_FRICTION_SEMI_IMPLICIT:
          TracerSemiImplicitBedFrictionRoughness(context, state, mannings_n, Fsum_x, Fsum_y, bedx, bedy, &tbx, &tby, e, d);
          break;
        default:
          break;
      }
    }

    sources[0][i] = riemannf[0][i] + ext_src[0][i];
    sources[1][i] = riemannf[1][i] - bedx - tbx + ext_src[1][i];
    sources[2][i] = riemannf[2][i] - bedy - tby + ext_src[2][i];
    for (CeedInt j = 0; j < context->tracer_ndof; ++j) {
      sources[flow_ndof + j][i] = riemannf[flow_ndof + j][i] + (e[j] - d[j]) + ext_src[flow_ndof + j][i];
    }
  }
  return 0;
}

CEED_QFUNCTION(TracerSourcesWithoutBedFriction)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  return TracerSources(ctx, Q, in, out, TRACER_BED_FRICTION_NONE);
}

CEED_QFUNCTION(TracerSourcesWithSemiImplicitBedFriction)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  return TracerSources(ctx, Q, in, out, TRACER_BED_FRICTION_SEMI_IMPLICIT);
}

#pragma GCC diagnostic   pop
#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
#pragma clang diagnostic pop

#endif
