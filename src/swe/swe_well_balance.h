#ifndef SWE_WELL_BALANCE_H
#define SWE_WELL_BALANCE_H
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

/// @brief Add weighted contribution of cell-centered height to the vertex-centered eta
/// @param ctx [in] SWE context
/// @param Q [in] number of quadrature points
/// @param in  [in] array of input fields
///             - in[0]: geom[num_cells][3] - an array associating the z1, z2, z3 of cell vertices with each cell
///             - in[1]: q[num_cells][3] - an array associating a 3-DOF solution input state with each cell
/// @param out [out] eta at cell vertices
/// @return 0 on success, otherwise an error code
CEED_QFUNCTION(SWEEtaVertex)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  // inputs
  const CeedScalar(*geom)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];  // z1, z2, z3 of cell vertices
  const CeedScalar(*q)[CEED_Q_VLA]    = (const CeedScalar(*)[CEED_Q_VLA])in[1];  // DOFs per cell

  // outputs
  CeedScalar(*etavertex)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const SWEContext context = (SWEContext)ctx;
  const CeedScalar tiny_h  = context->tiny_h;

  for (CeedInt i = 0; i < Q; i++) {
    SWEState   state = {q[0][i], q[1][i], q[2][i]};
    CeedScalar z1    = geom[0][i];
    CeedScalar z2    = geom[1][i];
    CeedScalar z3    = geom[2][i];
    CeedScalar wt    = geom[3][i];

    CeedScalar h3 = z3 - (z1 + z2 + z3) / 3.0;
    CeedScalar h2;
    if (z2 < z3) {
      h2 = (z2 - z1) * (z2 - z1) / (3.0 * (z3 - z1));
    } else {
      h2 = h3;
    }

    CeedScalar eta_cell = 0.0;
    if (state.h <= tiny_h) {
      // dry bed case
      eta_cell = z1;
    } else {
      // wet bed case
      if (state.h >= h3) {
        // all vertices are submerged
        eta_cell = (z1 + z2 + z3) / 3.0 + state.h;
      } else if (state.h > 0.0 && state.h <= h2) {
        // only one vertex is submerged
        eta_cell = z1 + pow(3.0 * state.h * (z2 - z1) * (z3 - z1), 1.0 / 3.0);
      } else if (state.h > h2 && state.h < h3) {
        // two vertices are submerged
        CeedScalar b = z3 - 3.0 * z1;
        CeedScalar c = z1 * z2 + z1 * z1 - z3 * z2 - 3.0 * state.h * (z3 - z1);
        eta_cell     = 0.5 * (-b + pow(b * b - 4.0 * c, 0.5));
      }
    }

    etavertex[0][i] = eta_cell * wt;
  }

  return 0;
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop

#endif  // SWE_WELL_BALANCE_H