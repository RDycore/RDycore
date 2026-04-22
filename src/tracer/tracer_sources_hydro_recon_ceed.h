#ifndef TRACER_SOURCES_HYDRO_RECON_CEED_H
#define TRACER_SOURCES_HYDRO_RECON_CEED_H

#include "tracer_sources_ceed.h"

// The following Q functions use C99 VLA features for shaping multidimensional
// arrays, which don't have the same drawbacks as VLA allocations.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla"

// Tracer source Q-function for hydrostatic reconstruction (HR).
//
// Under HR, the bed-slope source terms (-g*h*dz/dx, -g*h*dz/dy) are already
// accounted for by the pressure correction in the flux operator, so they are
// omitted here. Friction and sediment erosion/deposition source terms remain
// unchanged but receive bedx = bedy = 0.
//
// Input fields (same layout as TracerSources for operator compatibility):
//   in[0]: geom[num_owned_cells][2]            — dz/dx, dz/dy (not used)
//   in[1]: ext_src[num_owned_cells][num_comp]  — external sources
//   in[2]: mat_props[num_owned_cells][N]       — material properties
//   in[3]: riemannf[num_owned_cells][num_comp] — flux divergence
//   in[4]: q[num_owned_cells][num_comp]        — state variables
//
// Output fields:
//   out[0]: sources[num_owned_cells][num_comp]
CEED_QFUNCTION_HELPER int TracerSourcesHydroRecon(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[],
                                                  TracerBedFrictionType bed_friction_type) {
  // inputs (geom is in[0] but unused — bed slope handled by flux correction)
  const CeedScalar(*ext_src)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*mat_props)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  const CeedScalar(*riemannf)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  const CeedScalar(*q)[CEED_Q_VLA]         = (const CeedScalar(*)[CEED_Q_VLA])in[4];

  // outputs
  CeedScalar(*sources)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const TracerContext context   = (TracerContext)ctx;
  const CeedInt       flow_ndof = context->flow_ndof;
  const CeedScalar    tiny_h    = context->tiny_h;

  for (CeedInt i = 0; i < Q; i++) {
    TracerState state = {q[0][i], q[1][i], q[2][i]};
    for (CeedInt j = 0; j < context->tracer_ndof; ++j) {
      state.hci[j] = q[flow_ndof + j][i];
    }

    // bed slope terms are zero — already accounted for by HR flux correction
    const CeedScalar bedx = 0.0;
    const CeedScalar bedy = 0.0;

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
    sources[1][i] = riemannf[1][i] - tbx + ext_src[1][i];
    sources[2][i] = riemannf[2][i] - tby + ext_src[2][i];
    for (CeedInt j = 0; j < context->tracer_ndof; ++j) {
      sources[flow_ndof + j][i] = riemannf[flow_ndof + j][i] + (e[j] - d[j]) + ext_src[flow_ndof + j][i];
    }
  }

  return 0;
}

CEED_QFUNCTION(TracerSourcesHydroReconWithoutBedFriction)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  return TracerSourcesHydroRecon(ctx, Q, in, out, TRACER_BED_FRICTION_NONE);
}

CEED_QFUNCTION(TracerSourcesHydroReconWithSemiImplicitBedFriction)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  return TracerSourcesHydroRecon(ctx, Q, in, out, TRACER_BED_FRICTION_SEMI_IMPLICIT);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop

#endif
