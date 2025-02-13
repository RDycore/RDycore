#include <ceed/ceed.h>
#include <petscdmceed.h>
#include <private/rdycoreimpl.h>

// CEED uses C99 VLA features for shaping multidimensional
// arrays, which don't have the same drawbacks as VLA allocations.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"

#include "sediment_ceed_impl.h"

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop

static const PetscReal GRAVITY          = 9.806;   // gravitational acceleration [m/s^2]
static const PetscReal DENSITY_OF_WATER = 1000.0;  // [kg/m^3]

// frees a data context allocated using PETSc, returning a libCEED error code
static int FreeContextPetsc(void *data) {
  if (PetscFree(data)) return CeedError(NULL, CEED_ERROR_ACCESS, "PetscFree failed");
  return CEED_ERROR_SUCCESS;
}

// SWE + sediment dynamics Q function context
PetscErrorCode CreateSedimentQFunctionContext(Ceed ceed, const RDyConfig config, CeedQFunctionContext *qf_context) {
  PetscFunctionBeginUser;

  SedimentContext sediment_ctx;
  PetscCall(PetscCalloc1(1, &sediment_ctx));

  sediment_ctx->dtime                   = 0.0;
  sediment_ctx->tiny_h                  = config.physics.flow.tiny_h;
  sediment_ctx->gravity                 = GRAVITY;
  sediment_ctx->xq2018_threshold        = config.physics.flow.source.xq2018_threshold;
  sediment_ctx->kp_constant             = 0.001;
  sediment_ctx->settling_velocity       = 0.01;
  sediment_ctx->tau_critical_erosion    = 0.1;
  sediment_ctx->tau_critical_deposition = 1000.0;
  sediment_ctx->rhow                    = DENSITY_OF_WATER;
  sediment_ctx->sed_ndof                = config.physics.sediment.num_classes;
  sediment_ctx->flow_ndof               = 3;  // NOTE: SWE assumed!

  PetscCallCEED(CeedQFunctionContextCreate(ceed, qf_context));

  PetscCallCEED(CeedQFunctionContextSetData(*qf_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*sediment_ctx), sediment_ctx));

  PetscCallCEED(CeedQFunctionContextSetDataDestroy(*qf_context, CEED_MEM_HOST, FreeContextPetsc));

  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "time step", offsetof(struct SedimentContext_, dtime), 1, "Time step of TS"));

  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "small h value", offsetof(struct SedimentContext_, tiny_h), 1,
                                                   "Height threshold below which dry condition is assumed"));
  PetscCallCEED(
      CeedQFunctionContextRegisterDouble(*qf_context, "gravity", offsetof(struct SedimentContext_, gravity), 1, "Accelaration due to gravity"));

  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "xq2018_threshold", offsetof(struct SedimentContext_, xq2018_threshold), 1,
                                                   "Threshold for the treatment of Implicit XQ2018 method"));

  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "kp_constant", offsetof(struct SedimentContext_, kp_constant), 1,
                                                   "Krone-Partheniades erosion law constant [kg/m2/s]"));

  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "settling_velocity", offsetof(struct SedimentContext_, settling_velocity), 1,
                                                   "settling velocity of sediment class"));

  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "tau_critical_erosion", offsetof(struct SedimentContext_, tau_critical_erosion), 1,
                                                   "critical shear stress for erosion (N/m2)"));

  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "tau_critical_deposition", offsetof(struct SedimentContext_, tau_critical_deposition),
                                                   1, "critical shear stress for deposition (N/m2)"));

  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "rhow", offsetof(struct SedimentContext_, rhow), 1, "density of water"));

  PetscCallCEED(
      CeedQFunctionContextRegisterInt32(*qf_context, "sed_ndof", offsetof(struct SedimentContext_, sed_ndof), 1, "number of sediment classes"));
  PetscCallCEED(CeedQFunctionContextRegisterInt32(*qf_context, "flow_ndof", offsetof(struct SedimentContext_, flow_ndof), 1, "number of flow DoF"));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}
