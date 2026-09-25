#ifndef OPERATOR_IDENTITY_CEED_H
#define OPERATOR_IDENTITY_CEED_H

#include <ceed/types.h>

// we disable compiler warnings for implicitly-declared math functions known to
// the JIT compiler
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wimplicit-function-declaration"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-function-declaration"

// context for IdentityCopy: the number of components per element to copy
typedef struct IdentityCopyContext_ *IdentityCopyContext;
struct IdentityCopyContext_ {
  CeedInt num_comp;
};

// copies an "active" input field to an "active" output field unchanged.
//
// Used to refresh a passive field's backing vector from an active input
// vector via a real CeedOperator apply, instead of a standalone
// CeedElemRestrictionApply() call; see
// libceed-cuda-restriction-apply-broadcast-bug.md for why.
CEED_QFUNCTION(IdentityCopy)(void *ctx, CeedInt Q, const CeedScalar *const in[], CeedScalar *const out[]) {
  const IdentityCopyContext context  = (IdentityCopyContext)ctx;
  const CeedInt             num_comp = context->num_comp;
  const CeedScalar         *u        = in[0];
  CeedScalar               *v        = out[0];
  for (CeedInt k = 0; k < num_comp * Q; k++) v[k] = u[k];
  return CEED_ERROR_SUCCESS;
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop

#endif
