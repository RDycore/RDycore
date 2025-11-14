#ifndef SWE_SOURCES_CEED_H
#define SWE_SOURCES_CEED_H

#include "swe_types_ceed.h"

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

CEED_QFUNCTION(SWEFluxDivergenceSourceTerm)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const CeedScalar(*riemannf)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[3];  // riemann flux
  CeedScalar(*cell)[CEED_Q_VLA]            = (CeedScalar(*)[CEED_Q_VLA])out[0];

  for (CeedInt i = 0; i < Q; i++) {
    cell[0][i] += riemannf[0][i];
    cell[1][i] += riemannf[1][i];
    cell[2][i] += riemannf[2][i];
  }
  return 0;
}

CEED_QFUNCTION(SWEExternalSourceTerm)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const CeedScalar(*ext_src)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[1];  // external source (e.g. rain rate)
  CeedScalar(*cell)[CEED_Q_VLA]            = (CeedScalar(*)[CEED_Q_VLA])out[0];

  for (CeedInt i = 0; i < Q; i++) {
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
  const SWEContext context                 = (SWEContext)ctx;

  const CeedScalar gravity = context->gravity;

  for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar h     = q[0][i];

    const CeedScalar dz_dx = geom[0][i];
    const CeedScalar dz_dy = geom[1][i];

    const CeedScalar bedx = dz_dx * gravity * h;
    const CeedScalar bedy = dz_dy * gravity * h;

    cell[1][i] -= bedx;
    cell[2][i] -= bedy;
  }
  return 0;
}

// bed friction roughness -- must be at the end of the list of suboperators
CEED_QFUNCTION(SWEBedFrictionRoughnessSourceTermSemiImplicit)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const CeedScalar(*mat_props)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];  // material properties
  const CeedScalar(*q)[CEED_Q_VLA]         = (const CeedScalar(*)[CEED_Q_VLA])in[4];
  CeedScalar(*cell)[CEED_Q_VLA]            = (CeedScalar(*)[CEED_Q_VLA])out[0];
  const SWEContext context                 = (SWEContext)ctx;

  const CeedScalar dt      = context->dtime;
  const CeedScalar tiny_h  = context->tiny_h;
  const CeedScalar h_anuga = context->h_anuga_regular;
  const CeedScalar gravity = context->gravity;

  for (CeedInt i = 0; i < Q; i++) {
    SWEState         state = {q[0][i], q[1][i], q[2][i]};
    const CeedScalar h     = state.h;
    const CeedScalar hu    = state.hu;
    const CeedScalar hv    = state.hv;
    const CeedScalar denom = Square(h) + Square(h_anuga);

    const CeedScalar u = SafeDiv(state.hu * h, denom, h, tiny_h);
    const CeedScalar v = SafeDiv(state.hv * h, denom, h, tiny_h);

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
    }

    cell[1][i] -= tbx;
    cell[2][i] -= tby;
  }
  return 0;
}

/// @brief Adds contribution of the source-term using implicit time integration approach of:
///        Xia, Xilin, and Qiuhua Liang. "A new efficient implicit scheme for discretising the stiff
///        friction terms in the shallow water equations." Advances in water resources 117 (2018): 87-97.
///        https://www.sciencedirect.com/science/article/pii/S0309170818302124?ref=cra_js_challenge&fr=RR-1
///        (must be at the end of the list of suboperators)
CEED_QFUNCTION(SWEBedFrictionRoughnessSourceTermImplicitXQ2018)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const CeedScalar(*ext_src)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[1];  // external source (e.g. rain rate)
  const CeedScalar(*mat_props)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];  // material properties
  const CeedScalar(*q)[CEED_Q_VLA]         = (const CeedScalar(*)[CEED_Q_VLA])in[4];
  CeedScalar(*cell)[CEED_Q_VLA]            = (CeedScalar(*)[CEED_Q_VLA])out[0];
  const SWEContext context                 = (SWEContext)ctx;

  const CeedScalar dt               = context->dtime;
  const CeedScalar tiny_h           = context->tiny_h;
  const CeedScalar gravity          = context->gravity;
  const CeedScalar xq2018_threshold = context->xq2018_threshold;

  for (CeedInt i = 0; i < Q; i++) {
    SWEState         state = {q[0][i], q[1][i], q[2][i]};
    const CeedScalar h     = state.h;
    const CeedScalar hu    = state.hu;
    const CeedScalar hv    = state.hv;

    CeedScalar tbx = 0.0, tby = 0.0;
    if (h > tiny_h) {
      // gather other source contributions, since we're the last term
      const CeedScalar dsxdt = cell[1][i];
      const CeedScalar dsydt = cell[2][i];

      // defined in the text below equation 22 of XQ2018
      const CeedScalar Ax = dsxdt - ext_src[1][i]; // nominally Fsum_x - bedx
      const CeedScalar Ay = dsydt - ext_src[2][i]; // nominally Fsum_y - bedy

      // equation 27 of XQ2018
      const CeedScalar mx = hu + Ax * dt;
      const CeedScalar my = hv + Ay * dt;

      const CeedScalar mannings_n = mat_props[MATERIAL_PROPERTY_MANNINGS][i];
      const CeedScalar lambda     = gravity * Square(mannings_n) * pow(h, -4.0 / 3.0) * pow(Square(mx / h) + Square(my / h), 0.5);

      CeedScalar qx_nplus1 = 0.0, qy_nplus1 = 0.0;

      // equation 36 and 37 of XQ2018
      if (dt * lambda < xq2018_threshold) {
        qx_nplus1 = mx;
        qy_nplus1 = my;
      } else {
        qx_nplus1 = (mx - mx * pow(1.0 + 4.0 * dt * lambda, 0.5)) / (-2.0 * dt * lambda);
        qy_nplus1 = (my - my * pow(1.0 + 4.0 * dt * lambda, 0.5)) / (-2.0 * dt * lambda);
      }

      const CeedScalar q_magnitude = pow(Square(qx_nplus1) + Square(qy_nplus1), 0.5);

      // equation 21 and 22 of XQ2018
      tbx = gravity * Square(mannings_n) * pow(h, -7.0 / 3.0) * qx_nplus1 * q_magnitude;
      tby = gravity * Square(mannings_n) * pow(h, -7.0 / 3.0) * qy_nplus1 * q_magnitude;
    }

    cell[1][i] -= tbx;
    cell[2][i] -= tby;
  }
  return 0;
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop

#endif
