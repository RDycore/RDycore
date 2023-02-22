#include <private/rdycoreimpl.h>

PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec X, Vec F, void *ctx) {
  PetscFunctionBegin;

  RDy rdy = ctx;
  DM  dm  = rdy->dm;

  PetscFunctionReturn(0);
}
