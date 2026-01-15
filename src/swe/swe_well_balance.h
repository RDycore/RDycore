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

/// @brief Computes eta all cell centers, which is equal to water height + bed elevation if water height > tiny_h,
///        or just bed elevation otherwise.
/// @param ctx [in] SWE context
/// @param Q [in] number of quadrature points
/// @param in  [in] array of input fields
///             - in[0]: geom[num_cells][3] - an array associating the z1, z2, z3 of cell vertices with each cell
///             - in[1]: q[num_cells][3] - an array associating a 3-DOF solution input state with each cell
/// @param out [out] eta at cell centers
/// @return 0 on success, otherwise an error code
CEED_QFUNCTION(SWEEta)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  // inputs
  const CeedScalar(*geom)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];  // z1, z2, z3 of cell vertices
  const CeedScalar(*q)[CEED_Q_VLA]    = (const CeedScalar(*)[CEED_Q_VLA])in[1];  // riemann flux

  // outputs
  CeedScalar(*etacell)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const SWEContext context = (SWEContext)ctx;
  const CeedScalar tiny_h  = context->tiny_h;

  for (CeedInt i = 0; i < Q; i++) {
    SWEState   state = {q[0][i], q[1][i], q[2][i]};
    CeedScalar z1    = geom[0][i];
    CeedScalar z2    = geom[1][i];
    CeedScalar z3    = geom[2][i];

    if (state.h > tiny_h) {
      etacell[0][i] = (z1 + z2 + z3) / 3.0 + state.h;
    } else {
      etacell[0][i] = (z1 + z2 + z3) / 3.0;
    }
  }

  return 0;
}

CEED_QFUNCTION(SWEDelHAlongEdge)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  // inputs
  const CeedScalar(*geom)[CEED_Q_VLA]         = (const CeedScalar(*)[CEED_Q_VLA])in[0];  // z values at begin and end of edge
  const CeedScalar(*eta_vertices)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];  // eta at cell vertices

  // outputs
  CeedScalar(*delH_along_edge)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  for (CeedInt i = 0; i < Q; i++) {
    CeedScalar zbeg = geom[0][i];
    CeedScalar zend = geom[1][i];

    CeedScalar eta_beg = eta_vertices[0][i];
    CeedScalar eta_end = eta_vertices[1][i];

    CeedScalar h1 = eta_beg - zbeg;
    CeedScalar h2 = eta_end - zend;

    if (h1 < 0.0) h1 = 0.0;
    if (h2 < 0.0) h2 = 0.0;

    delH_along_edge[0][i] = h2 - h1;
  }

  return 0;
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop

#endif  // SWE_WELL_BALANCE_H