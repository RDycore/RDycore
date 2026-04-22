#ifndef SWE_SOURCES_HYDRO_RECON_CEED_H
#define SWE_SOURCES_HYDRO_RECON_CEED_H

#include "swe_sources_ceed.h"

// The following Q functions use C99 VLA features for shaping multidimensional
// arrays, which don't have the same drawbacks as VLA allocations.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla"

// SWE source Q-function for hydrostatic reconstruction (HR).
//
// Under HR, the bed-slope source terms (-g*h*dz/dx, -g*h*dz/dy) are already
// accounted for by the pressure correction in the flux operator, so they are
// omitted here. The friction source terms remain unchanged but receive
// bedx = bedy = 0, since the bed-slope contribution is already embedded in
// the flux divergence (riemannf).
//
// Input fields (same layout as SWESources for operator compatibility):
//   in[0]: geom[num_owned_cells][2]       — dz/dx, dz/dy (not used)
//   in[1]: ext_src[num_owned_cells][3]    — external sources
//   in[2]: mat_props[num_owned_cells][N]  — material properties
//   in[3]: riemannf[num_owned_cells][3]   — flux divergence
//   in[4]: q[num_owned_cells][3]          — state variables
//
// Output fields:
//   out[0]: sources[num_owned_cells][3]
CEED_QFUNCTION_HELPER int SWESourcesHydroRecon(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[],
                                               SWEBedFrictionType bed_friction_type) {
  // inputs (geom is in[0] but unused — bed slope handled by flux correction)
  const CeedScalar(*ext_src)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*mat_props)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  const CeedScalar(*riemannf)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  const CeedScalar(*q)[CEED_Q_VLA]         = (const CeedScalar(*)[CEED_Q_VLA])in[4];

  // outputs
  CeedScalar(*sources)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const SWEContext context = (SWEContext)ctx;
  const CeedScalar tiny_h  = context->tiny_h;

  for (CeedInt i = 0; i < Q; i++) {
    SWEState state = {q[0][i], q[1][i], q[2][i]};

    // bed slope terms are zero — already accounted for by HR flux correction
    const CeedScalar bedx = 0.0;
    const CeedScalar bedy = 0.0;

    CeedScalar tbx = 0.0, tby = 0.0;
    if (state.h > tiny_h) {
      const CeedScalar Fsum_x     = riemannf[1][i];
      const CeedScalar Fsum_y     = riemannf[2][i];
      const CeedScalar mannings_n = mat_props[MATERIAL_PROPERTY_MANNINGS][i];

      switch (bed_friction_type) {
        case SWE_BED_FRICTION_SEMI_IMPLICIT:
          SWESemiImplicitBedFrictionRoughness(context, state, mannings_n, Fsum_x, Fsum_y, bedx, bedy, &tbx, &tby);
          break;
        case SWE_BED_FRICTION_IMPLICIT_XQ2018:
          SWEImplicitBedFrictionRoughnessXQ2018(context, state, mannings_n, Fsum_x, Fsum_y, bedx, bedy, &tbx, &tby);
          break;
        default:
          break;
      }
    }

    sources[0][i] = riemannf[0][i] + ext_src[0][i];
    sources[1][i] = riemannf[1][i] - tbx + ext_src[1][i];
    sources[2][i] = riemannf[2][i] - tby + ext_src[2][i];
  }

  return 0;
}

CEED_QFUNCTION(SWESourcesHydroReconWithoutBedFriction)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  return SWESourcesHydroRecon(ctx, Q, in, out, SWE_BED_FRICTION_NONE);
}

CEED_QFUNCTION(SWESourcesHydroReconWithSemiImplicitBedFriction)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  return SWESourcesHydroRecon(ctx, Q, in, out, SWE_BED_FRICTION_SEMI_IMPLICIT);
}

CEED_QFUNCTION(SWESourcesHydroReconWithImplicitBedFrictionXQ2018)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  return SWESourcesHydroRecon(ctx, Q, in, out, SWE_BED_FRICTION_IMPLICIT_XQ2018);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop

#endif
