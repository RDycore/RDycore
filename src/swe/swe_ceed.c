#include <ceed/ceed.h>
#include <petscdmceed.h>
#include <private/rdycoreimpl.h>

// CEED uses C99 VLA features for shaping multidimensional
// arrays, which don't have the same drawbacks as VLA allocations.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"

#include "swe_types_ceed.h"

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop

// frees a data context allocated using PETSc, returning a libCEED error code
static int FreeContextPetsc(void *data) {
  if (PetscFree(data)) return CeedError(NULL, CEED_ERROR_ACCESS, "PetscFree failed");
  return CEED_ERROR_SUCCESS;
}

// SWE-only Q function context
PetscErrorCode CreateSWEQFunctionContext(Ceed ceed, const RDyConfig config, CeedQFunctionContext *qf_context) {
  PetscFunctionBeginUser;

  SWEContext swe_ctx;
  PetscCall(PetscCalloc1(1, &swe_ctx));

  swe_ctx->dtime            = 0.0;
  swe_ctx->tiny_h           = config.physics.flow.tiny_h;
  swe_ctx->gravity          = 9.806;
  swe_ctx->xq2018_threshold = config.physics.flow.source.xq2018_threshold;
  swe_ctx->h_anuga_regular  = config.physics.flow.h_anuga_regular;

  PetscCallCEED(CeedQFunctionContextCreate(ceed, qf_context));
  PetscCallCEED(CeedQFunctionContextSetData(*qf_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*swe_ctx), swe_ctx));
  PetscCallCEED(CeedQFunctionContextSetDataDestroy(*qf_context, CEED_MEM_HOST, FreeContextPetsc));
  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "time step", offsetof(struct SWEContext_, dtime), 1, "Time step of TS"));
  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "small h value", offsetof(struct SWEContext_, tiny_h), 1,
                                                   "Height threshold below which dry condition is assumed"));
  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "h_anuga_regular", offsetof(struct SWEContext_, h_anuga_regular), 1,
                                                   "ANUGA height parameter for velocity regularization"));
  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "gravity", offsetof(struct SWEContext_, gravity), 1, "Accelaration due to gravity"));
  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "xq2018_threshold", offsetof(struct SWEContext_, xq2018_threshold), 1,
                                                   "Threshold for the treatment of Implicit XQ2018 method"));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}
