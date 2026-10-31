#ifndef SWE_FLUXES_AUGMENTED_ROE_CEED_H
#define SWE_FLUXES_AUGMENTED_ROE_CEED_H

#include "swe_augmented_roe_flux_ceed.h"
#include "swe_interior_geom_ceed.h"
#include "swe_types_ceed.h"

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

// SWE interior flux Q-function with the augmented Roe solver, whose
// fluctuations include the bed-slope source. The left and right cells see
// different fluxes (see SWERiemannFlux_AugmentedRoe).
//
// Input fields (order determines indexing):
//   in[0]: geom[num_interior_edges][HR_NUM_COMP_INTERIOR_GEOM]
//     see HRInteriorGeomIndex for component layout
//   in[1]: q_left[num_interior_edges][3]
//   in[2]: q_right[num_interior_edges][3]
//
// Output fields:
//   out[0]: cell_left[num_interior_edges][3]
//   out[1]: cell_right[num_interior_edges][3]
//   out[2]: flux[num_interior_edges][3] (flux seen by the left cell, whose
//           mass component is the edge mass flux)
//   out[3]: courant_number[num_interior_edges][2]
CEED_QFUNCTION(SWEFlux_AugmentedRoe)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const CeedScalar(*geom)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_L)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*q_R)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  CeedScalar(*cell_L)[CEED_Q_VLA]      = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*cell_R)[CEED_Q_VLA]      = (CeedScalar(*)[CEED_Q_VLA])out[1];
  CeedScalar(*flux_out)[CEED_Q_VLA]    = (CeedScalar(*)[CEED_Q_VLA])out[2];
  CeedScalar(*courant_num)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[3];
  const SWEContext context             = (SWEContext)ctx;

  const CeedScalar dt      = context->dtime;
  const CeedScalar tiny_h  = context->tiny_h;
  const CeedScalar h_anuga = context->h_anuga_regular;
  const CeedScalar gravity = context->gravity;

  for (CeedInt i = 0; i < Q; i++) {
    SWEState qL = {q_L[0][i], q_L[1][i], q_L[2][i]};
    SWEState qR = {q_R[0][i], q_R[1][i], q_R[2][i]};

    if (qL.h > tiny_h || qR.h > tiny_h) {
      CeedScalar flux_L[3], flux_R[3], amax;
      SWERiemannFlux_AugmentedRoe(gravity, tiny_h, h_anuga, qL, qR, geom[HR_INTERIOR_ZC_LEFT][i], geom[HR_INTERIOR_ZC_RIGHT][i],
                                  geom[HR_INTERIOR_SN][i], geom[HR_INTERIOR_CN][i], flux_L, flux_R, &amax);
      for (CeedInt j = 0; j < 3; j++) {
        cell_L[j][i]   = flux_L[j] * geom[HR_INTERIOR_NEG_L_OVER_AL][i];
        cell_R[j][i]   = flux_R[j] * geom[HR_INTERIOR_L_OVER_AR][i];
        flux_out[j][i] = flux_L[j];
      }
      courant_num[0][i] = -amax * geom[HR_INTERIOR_NEG_L_OVER_AL][i] * dt;
      courant_num[1][i] = amax * geom[HR_INTERIOR_L_OVER_AR][i] * dt;
    } else {
      for (CeedInt j = 0; j < 3; j++) {
        cell_L[j][i]   = 0.0;
        cell_R[j][i]   = 0.0;
        flux_out[j][i] = 0.0;
      }
      courant_num[0][i] = 0.0;
      courant_num[1][i] = 0.0;
    }
  }
  return 0;
}

#pragma GCC diagnostic   pop
#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
#pragma clang diagnostic pop

#endif
