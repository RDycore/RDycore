#ifndef TRACER_FLUXES_HYDRO_RECON_CEED_H
#define TRACER_FLUXES_HYDRO_RECON_CEED_H

#include "tracer_types_ceed.h"

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

#include "tracer_roe_flux_ceed.h"

// The following Q functions use C99 VLA features for shaping multidimensional
// arrays, which don't have the same drawbacks as VLA allocations.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla"

// Tracer interior flux Q-function with hydrostatic reconstruction
//
// Input fields (order determines indexing):
//   in[0]: geom[num_interior_edges][HR_NUM_COMP_INTERIOR_GEOM]
//     see HRInteriorGeomIndex for component layout
//   in[1]: q_left[num_interior_edges][num_comp]
//   in[2]: q_right[num_interior_edges][num_comp]
//
// Output fields:
//   out[0]: cell_left[num_interior_edges][num_comp]
//   out[1]: cell_right[num_interior_edges][num_comp]
//   out[2]: courant_number[num_interior_edges][2]
CEED_QFUNCTION_HELPER int TracerFlux_HydroRecon(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[],
                                                RiemannFluxType flux_type) {
  const CeedScalar(*geom)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_L)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*q_R)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  CeedScalar(*cell_L)[CEED_Q_VLA]      = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*cell_R)[CEED_Q_VLA]      = (CeedScalar(*)[CEED_Q_VLA])out[1];
  CeedScalar(*courant_num)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[2];
  const TracerContext context          = (TracerContext)ctx;

  const CeedScalar dt      = context->dtime;
  const CeedScalar tiny_h  = context->tiny_h;
  const CeedScalar h_anuga = context->h_anuga_regular;
  const CeedScalar gravity = context->gravity;

  const CeedInt flow_ndof   = context->flow_ndof;
  const CeedInt tracer_ndof = context->tracer_ndof;
  const CeedInt tot_ndof    = flow_ndof + tracer_ndof;

  for (CeedInt i = 0; i < Q; i++) {
    TracerState qL = {q_L[0][i], q_L[1][i], q_L[2][i]};
    TracerState qR = {q_R[0][i], q_R[1][i], q_R[2][i]};
    for (CeedInt j = 0; j < tracer_ndof; ++j) {
      qL.hci[j] = q_L[flow_ndof + j][i];
      qR.hci[j] = q_R[flow_ndof + j][i];
    }

    CeedScalar zc_L = geom[HR_INTERIOR_ZC_LEFT][i];
    CeedScalar zc_R = geom[HR_INTERIOR_ZC_RIGHT][i];

    // hydrostatic reconstruction:
    //   eta_L = h_L + z_L,  eta_R = h_R + z_R
    //   z_max = max(z_L, z_R)
    //   h_L*  = max(0, eta_L - z_max)
    //   h_R*  = max(0, eta_R - z_max)
    CeedScalar eta_L  = qL.h + zc_L;
    CeedScalar eta_R  = qR.h + zc_R;
    CeedScalar z_max  = fmax(zc_L, zc_R);
    CeedScalar hL_rec = fmax(0.0, eta_L - z_max);
    CeedScalar hR_rec = fmax(0.0, eta_R - z_max);

    // preserve velocities: reconstruct momentum from reconstructed h
    CeedScalar denom_L = Square(qL.h) + Square(h_anuga);
    CeedScalar denom_R = Square(qR.h) + Square(h_anuga);
    CeedScalar uL      = SafeDiv(qL.hu * qL.h, denom_L, qL.h, tiny_h);
    CeedScalar vL      = SafeDiv(qL.hv * qL.h, denom_L, qL.h, tiny_h);
    CeedScalar uR      = SafeDiv(qR.hu * qR.h, denom_R, qR.h, tiny_h);
    CeedScalar vR      = SafeDiv(qR.hv * qR.h, denom_R, qR.h, tiny_h);

    TracerState qL_rec = {hL_rec, hL_rec * uL, hL_rec * vL};
    TracerState qR_rec = {hR_rec, hR_rec * uR, hR_rec * vR};

    // preserve concentrations: reconstruct hci from reconstructed h
    for (CeedInt j = 0; j < tracer_ndof; ++j) {
      CeedScalar ciL = SafeDiv(qL.hci[j], qL.h, qL.h, tiny_h);
      CeedScalar ciR = SafeDiv(qR.hci[j], qR.h, qR.h, tiny_h);
      qL_rec.hci[j]  = hL_rec * ciL;
      qR_rec.hci[j]  = hR_rec * ciR;
    }

    if (qL.h > tiny_h || qR.h > tiny_h) {
      CeedScalar flux[MAX_NUM_FIELD_COMPONENTS] = {0};
      CeedScalar amax                           = 0.0;

      // Guard the Riemann solver on the RECONSTRUCTED heights: HR can drive
      // both hL_rec and hR_rec to zero even when the original h is positive
      // (e.g. water surface below the neighbor's bed elevation).  Calling the
      // Roe solver with both heights == 0 causes 0/0 = NaN in the Roe
      // averages.
      if (hL_rec > tiny_h || hR_rec > tiny_h) {
        switch (flux_type) {
          case RIEMANN_FLUX_ROE:
            TracerRiemannFlux_Roe(gravity, tiny_h, qL_rec, qR_rec, geom[HR_INTERIOR_SN][i], geom[HR_INTERIOR_CN][i], flow_ndof, tracer_ndof, flux,
                                  &amax);
            break;
        }
      }

      for (CeedInt j = 0; j < tot_ndof; j++) {
        cell_L[j][i] = flux[j] * geom[HR_INTERIOR_NEG_L_OVER_AL][i];
        cell_R[j][i] = flux[j] * geom[HR_INTERIOR_L_OVER_AR][i];
      }

      // hydrostatic pressure correction applied separately to momentum
      // components only:
      //   corr_L = g/2 * (h_L^2 - h_L*^2)  added to left cell only
      //   corr_R = g/2 * (h_R^2 - h_R*^2)  added to right cell only
      CeedScalar corr_L = 0.5 * gravity * (Square(qL.h) - Square(hL_rec));
      CeedScalar corr_R = 0.5 * gravity * (Square(qR.h) - Square(hR_rec));
      cell_L[1][i] += corr_L * geom[HR_INTERIOR_CN][i] * geom[HR_INTERIOR_NEG_L_OVER_AL][i];
      cell_L[2][i] += corr_L * geom[HR_INTERIOR_SN][i] * geom[HR_INTERIOR_NEG_L_OVER_AL][i];
      cell_R[1][i] += corr_R * geom[HR_INTERIOR_CN][i] * geom[HR_INTERIOR_L_OVER_AR][i];
      cell_R[2][i] += corr_R * geom[HR_INTERIOR_SN][i] * geom[HR_INTERIOR_L_OVER_AR][i];
      courant_num[0][i] = -amax * geom[HR_INTERIOR_NEG_L_OVER_AL][i] * dt;
      courant_num[1][i] = amax * geom[HR_INTERIOR_L_OVER_AR][i] * dt;
    } else {
      for (CeedInt j = 0; j < tot_ndof; j++) {
        cell_L[j][i] = 0.0;
        cell_R[j][i] = 0.0;
      }
      courant_num[0][i] = 0.0;
      courant_num[1][i] = 0.0;
    }
  }
  return 0;
}

CEED_QFUNCTION(TracerFlux_HydroRecon_Roe)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  return TracerFlux_HydroRecon(ctx, Q, in, out, RIEMANN_FLUX_ROE);
}

#pragma GCC diagnostic   pop
#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
#pragma clang diagnostic pop

#endif
