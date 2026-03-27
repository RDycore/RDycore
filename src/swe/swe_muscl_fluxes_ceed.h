#ifndef SWE_MUSCL_FLUXES_CEED_H
#define SWE_MUSCL_FLUXES_CEED_H

// MUSCL-reconstructed interior flux Q-function for SWE.
// The host pre-computes face-reconstructed states and writes them into
// q_reconstructed before each CeedOperatorApply call.  This Q-function
// simply reads those pre-built values (in[1]/in[2]) and feeds them straight
// into the Roe Riemann solver — identical to SWEFlux_Roe except for the
// field names and the h >= 0 clamp applied after reading.

#include <math.h>

#include "swe_types_ceed.h"

#ifndef Square
#define Square(x) ((x) * (x))
#endif
#ifndef SafeDiv
#define SafeDiv(a, b, c, tiny) ((c) > (tiny) ? (a) / (b) : 0.0)
#endif

CEED_QFUNCTION_HELPER CeedScalar ComputeDhv_MUSCL(CeedScalar zv_beg, CeedScalar zv_end, CeedScalar eta_beg, CeedScalar eta_end) {
  CeedScalar hv_beg = fmax(eta_beg - zv_beg, 0.0);
  CeedScalar hv_end = fmax(eta_end - zv_end, 0.0);
  return hv_end - hv_beg;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wimplicit-function-declaration"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-function-declaration"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla"

#include "swe_roe_flux_ceed.h"
typedef enum {
  RIEMANN_FLUX_ROE_MUSCL,
} RiemannFluxType_MUSCL;

// Interior flux Q-function: reads pre-reconstructed face states from in[1]/in[2].
CEED_QFUNCTION_HELPER int SWEFlux_MUSCL(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[], RiemannFluxType_MUSCL flux_type) {
  // in[0]: geom[6]         — sn, cn, -L/A_l, L/A_r, z_beg_vertex, z_end_vertex
  // in[1]: q_left_face[3]  — pre-reconstructed h, hu, hv at face (from left cell)
  // in[2]: q_right_face[3] — pre-reconstructed h, hu, hv at face (from right cell)
  // in[3]: eta_vert_beg[1] — water surface elevation at edge start vertex
  // in[4]: eta_vert_end[1] — water surface elevation at edge end vertex
  const CeedScalar(*geom)[CEED_Q_VLA]          = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_left_face)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*q_right_face)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  const CeedScalar(*eta_vert_beg)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  const CeedScalar(*eta_vert_end)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[4];
  CeedScalar(*cell_L)[CEED_Q_VLA]              = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*cell_R)[CEED_Q_VLA]              = (CeedScalar(*)[CEED_Q_VLA])out[1];
  CeedScalar(*courant_num)[CEED_Q_VLA]         = (CeedScalar(*)[CEED_Q_VLA])out[2];
  const SWEContext context                     = (SWEContext)ctx;

  const CeedScalar dt      = context->dtime;
  const CeedScalar tiny_h  = context->tiny_h;
  const CeedScalar h_anuga = context->h_anuga_regular;
  const CeedScalar gravity = context->gravity;

  for (CeedInt i = 0; i < Q; i++) {
    // Clamp h from below — linear reconstruction can produce small negatives
    SWEState qL = {fmax(0.0, q_left_face[0][i]),  q_left_face[1][i],  q_left_face[2][i]};
    SWEState qR = {fmax(0.0, q_right_face[0][i]), q_right_face[1][i], q_right_face[2][i]};

    CeedScalar dhv = ComputeDhv_MUSCL(geom[4][i], geom[5][i], eta_vert_beg[0][i], eta_vert_end[0][i]);

    CeedScalar flux[3], amax;

    if (qL.h > tiny_h || qR.h > tiny_h) {
      switch (flux_type) {
        case RIEMANN_FLUX_ROE_MUSCL:
          SWERiemannFlux_Roe(gravity, tiny_h, h_anuga, qL, qR, geom[0][i], geom[1][i], dhv, flux, &amax);
          break;
      }
      for (CeedInt j = 0; j < 3; j++) {
        cell_L[j][i] = flux[j] * geom[2][i];
        cell_R[j][i] = flux[j] * geom[3][i];
      }
      courant_num[0][i] = -amax * geom[2][i] * dt;
      courant_num[1][i] =  amax * geom[3][i] * dt;
    } else {
      for (CeedInt j = 0; j < 3; j++) {
        cell_L[j][i] = 0.0;
        cell_R[j][i] = 0.0;
      }
      courant_num[0][i] = 0.0;
      courant_num[1][i] = 0.0;
    }
  }
  return 0;
}

CEED_QFUNCTION(SWEFlux_Roe_MUSCL)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  return SWEFlux_MUSCL(ctx, Q, in, out, RIEMANN_FLUX_ROE_MUSCL);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
#pragma GCC diagnostic   pop
#pragma clang diagnostic pop

// SWEFlux_Roe_MUSCL_loc is defined automatically by CEED_QFUNCTION(SWEFlux_Roe_MUSCL)
// above as __FILE__ ":" "SWEFlux_Roe_MUSCL" — do not redefine it here.

#endif
