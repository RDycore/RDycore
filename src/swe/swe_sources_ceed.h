#ifndef SWE_SOURCES_CEED_H
#define SWE_SOURCES_CEED_H

#include "swe_types_ceed.h"

#ifndef Square
#define Square(x) ((x) * (x))
#endif
#ifndef SafeDiv
#define SafeDiv(a, b, c, tiny) ((c) > (tiny) ? (a) / (b) : 0.0)
#endif

// supported bed friction source term methods
typedef enum {
  SWE_BED_FRICTION_NONE,
  SWE_BED_FRICTION_SEMI_IMPLICIT,
  SWE_BED_FRICTION_IMPLICIT_XQ2018,
} SWEBedFrictionType;

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

// semi-implicit bed friction roughness term
int CEED_QFUNCTION_HELPER SWESemiImplicitBedFrictionRoughness(SWEContext context, SWEState state, CeedScalar mannings_n, CeedScalar Fsum_x,
                                                              CeedScalar Fsum_y, CeedScalar bedx, CeedScalar bedy, CeedScalar *tbx, CeedScalar *tby) {
  const CeedScalar dt      = context->dtime;
  const CeedScalar tiny_h  = context->tiny_h;
  const CeedScalar h_anuga = context->h_anuga_regular;
  const CeedScalar gravity = context->gravity;

  const CeedScalar h     = state.h;
  const CeedScalar hu    = state.hu;
  const CeedScalar hv    = state.hv;
  const CeedScalar denom = Square(h) + Square(h_anuga);

  const CeedScalar u = SafeDiv(state.hu * h, denom, h, tiny_h);
  const CeedScalar v = SafeDiv(state.hv * h, denom, h, tiny_h);

  const CeedScalar Cd       = gravity * Square(mannings_n) * pow(h, -1.0 / 3.0);
  const CeedScalar velocity = sqrt(Square(u) + Square(v));

  const CeedScalar tb     = Cd * velocity / h;
  const CeedScalar factor = tb / (1.0 + dt * tb);

  *tbx = (hu + dt * (Fsum_x - bedx)) * factor;
  *tby = (hv + dt * (Fsum_y - bedy)) * factor;

  return 0;
}

/// @brief Adds contribution of the source-term using implicit time integration approach of:
///        Xia, Xilin, and Qiuhua Liang. "A new efficient implicit scheme for discretising the stiff
///        friction terms in the shallow water equations." Advances in water resources 117 (2018): 87-97.
///        https://www.sciencedirect.com/science/article/pii/S0309170818302124?ref=cra_js_challenge&fr=RR-1
///        (must be at the end of the list of suboperators)
int CEED_QFUNCTION_HELPER SWEImplicitBedFrictionRoughnessXQ2018(SWEContext context, SWEState state, CeedScalar mannings_n, CeedScalar Fsum_x,
                                                                CeedScalar Fsum_y, CeedScalar bedx, CeedScalar bedy, CeedScalar *tbx,
                                                                CeedScalar *tby) {
  const CeedScalar dt               = context->dtime;
  const CeedScalar gravity          = context->gravity;
  const CeedScalar xq2018_threshold = context->xq2018_threshold;

  // defined in the text below equation 22 of XQ2018
  const CeedScalar Ax = Fsum_x - bedx;
  const CeedScalar Ay = Fsum_y - bedy;

  // equation 27 of XQ2018
  const CeedScalar mx = state.hu + Ax * dt;
  const CeedScalar my = state.hv + Ay * dt;

  const CeedScalar lambda = gravity * Square(mannings_n) * pow(state.h, -4.0 / 3.0) * pow(Square(mx / state.h) + Square(my / state.h), 0.5);

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
  *tbx = gravity * Square(mannings_n) * pow(state.h, -7.0 / 3.0) * qx_nplus1 * q_magnitude;
  *tby = gravity * Square(mannings_n) * pow(state.h, -7.0 / 3.0) * qy_nplus1 * q_magnitude;

  return 0;
}

CEED_QFUNCTION_HELPER int SWESources(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[],
                                     SWEBedFrictionType bed_friction_type) {
  // inputs
  const CeedScalar(*geom)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[0];  // dz/dx, dz/dy
  const CeedScalar(*ext_src)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[1];  // external source (e.g. rain rate)
  const CeedScalar(*mat_props)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];  // material properties
  const CeedScalar(*riemannf)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[3];  // riemann flux
  const CeedScalar(*q)[CEED_Q_VLA]         = (const CeedScalar(*)[CEED_Q_VLA])in[4];  // state variables

  // outputs
  CeedScalar(*sources)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const SWEContext context = (SWEContext)ctx;
  const CeedScalar tiny_h  = context->tiny_h;
  const CeedScalar gravity = context->gravity;

  for (CeedInt i = 0; i < Q; i++) {
    SWEState state = {q[0][i], q[1][i], q[2][i]};

    const CeedScalar dz_dx = geom[0][i];
    const CeedScalar dz_dy = geom[1][i];

    const CeedScalar bedx = dz_dx * gravity * state.h;
    const CeedScalar bedy = dz_dy * gravity * state.h;

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
    sources[1][i] = riemannf[1][i] - bedx - tbx + ext_src[1][i];
    sources[2][i] = riemannf[2][i] - bedy - tby + ext_src[2][i];
  }

  return 0;
}

CEED_QFUNCTION(SWESourcesWithoutBedFriction)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  return SWESources(ctx, Q, in, out, SWE_BED_FRICTION_NONE);
}

CEED_QFUNCTION(SWESourcesWithSemiImplicitBedFriction)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  return SWESources(ctx, Q, in, out, SWE_BED_FRICTION_SEMI_IMPLICIT);
}

CEED_QFUNCTION(SWESourcesWithImplicitBedFrictionXQ2018)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  return SWESources(ctx, Q, in, out, SWE_BED_FRICTION_IMPLICIT_XQ2018);

// IJacobian Q function for implicit treatment of bed friction source term sf
CEED_QFUNCTION(SWEIJacobian_IMEX)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  // inputs
  const CeedScalar(*mat_props)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];  // material properties
  const CeedScalar(*q)[CEED_Q_VLA]         = (const CeedScalar(*)[CEED_Q_VLA])in[1];  // solution

  // outputs (recall arrays are stored in column-major order!)
  CeedScalar(*J)[3][CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];  // 3x3 block jacobian

  const SWEContext context = (SWEContext)ctx;

  const CeedScalar dt      = context->dtime;
  const CeedScalar tiny_h  = context->tiny_h;
  const CeedScalar h_anuga = context->h_anuga_regular;
  const CeedScalar gravity = context->gravity;

  for (CeedInt i = 0; i < Q; i++) {
    SWEState         state = {q[0][i], q[1][i], q[2][i]};
    const CeedScalar h     = state.h;
    const CeedScalar denom = Square(h) + Square(h_anuga);

    const CeedScalar u = SafeDiv(state.hu * h, denom, h, tiny_h);
    const CeedScalar v = SafeDiv(state.hv * h, denom, h, tiny_h);

    const CeedScalar mannings_n = mat_props[MATERIAL_PROPERTY_MANNINGS][i];
    const CeedScalar n2         = Square(mannings_n);

    // velocity magnitude
    CeedScalar velocity = sqrt(Square(u) + Square(v));

    // bed friction partial derivatives
    CeedScalar dsfdu[3][3] = {0};

    // u-component derivatives
    dsfdu[0][1] = -1.0 / 3.0 * gravity * n2 * pow(h, -4.0 / 3.0) * u * velocity;  // d/dh
    dsfdu[1][1] = gravity * n2 * pow(h, -1.0 / 3.0) * (velocity + u / velocity);  // d/du
    dsfdu[2][1] = gravity * n2 * pow(h, -1.0 / 3.0) * u * v / velocity;           // d/dv

    // v-component derivatives
    dsfdu[0][2] = -1.0 / 3.0 * gravity * n2 * pow(h, -4.0 / 3.0) * v * velocity;  // d/dh
    dsfdu[1][2] = gravity * n2 * pow(h, -1.0 / 3.0) * u * v / velocity;           // d/du
    dsfdu[2][2] = gravity * n2 * pow(h, -1.0 / 3.0) * (velocity + v / velocity);  // d/dv

    // jacobian
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        J[j][k][i] = -dt * dsfdu[j][k];
      }
    }
    J[0][0][i] += 1.0;
    J[1][1][i] += 1.0;
    J[2][2][i] += 1.0;
  }
  return 0;
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop

#endif
