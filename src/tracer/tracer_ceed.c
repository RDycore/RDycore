#include <ceed/ceed.h>
#include <petscdmceed.h>
#include <private/rdycoreimpl.h>

// CEED uses C99 VLA features for shaping multidimensional
// arrays, which don't have the same drawbacks as VLA allocations.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"

#include "tracer_types_ceed.h"

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop

static const PetscReal GRAVITY          = 9.806;   // gravitational acceleration [m/s^2]
static const PetscReal DENSITY_OF_WATER = 1000.0;  // [kg/m^3]

// frees a data context allocated using PETSc, returning a libCEED error code
static int FreeContextPetsc(void *data) {
  if (PetscFree(data)) return CeedError(NULL, CEED_ERROR_ACCESS, "PetscFree failed");
  return CEED_ERROR_SUCCESS;
}

// SWE + tracers dynamics Q function context
PetscErrorCode CreateTracersQFunctionContext(Ceed ceed, const RDyConfig config, CeedQFunctionContext *qf_context) {
  PetscFunctionBeginUser;

  TracersContext tracers_ctx;
  PetscCall(PetscCalloc1(1, &tracers_ctx));

  PetscInt num_tracers = config.physics.sediment.num_classes +
                         (config.physics.salinity ? 1 : 0) +
                         (config.physics.heat ? 1 : 0);

  tracers_ctx->dtime                   = 0.0;
  tracers_ctx->tiny_h                  = config.physics.flow.tiny_h;
  tracers_ctx->gravity                 = GRAVITY;
  tracers_ctx->xq2018_threshold        = config.physics.flow.source.xq2018_threshold;
  tracers_ctx->kp_constant             = 0.001;
  tracers_ctx->settling_velocity       = 0.01;
  tracers_ctx->tau_critical_erosion    = 0.1;
  tracers_ctx->tau_critical_deposition = 1000.0;
  tracers_ctx->rhow                    = DENSITY_OF_WATER;
  tracers_ctx->tracers_ndof            = num_tracers;
  tracers_ctx->flow_ndof               = 3;  // NOTE: SWE assumed!

  PetscCallCEED(CeedQFunctionContextCreate(ceed, qf_context));

  PetscCallCEED(CeedQFunctionContextSetData(*qf_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*tracers_ctx), tracers_ctx));

  PetscCallCEED(CeedQFunctionContextSetDataDestroy(*qf_context, CEED_MEM_HOST, FreeContextPetsc));

  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "time step", offsetof(struct TracersContext_, dtime), 1, "Time step of TS"));

  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "small h value", offsetof(struct TracersContext_, tiny_h), 1,
                                                   "Height threshold below which dry condition is assumed"));
  PetscCallCEED(
      CeedQFunctionContextRegisterDouble(*qf_context, "gravity", offsetof(struct TracersContext_, gravity), 1, "Accelaration due to gravity"));

  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "xq2018_threshold", offsetof(struct TracersContext_, xq2018_threshold), 1,
                                                   "Threshold for the treatment of Implicit XQ2018 method"));

  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "kp_constant", offsetof(struct TracersContext_, kp_constant), 1,
                                                   "Krone-Partheniades erosion law constant [kg/m2/s]"));

  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "settling_velocity", offsetof(struct TracersContext_, settling_velocity), 1,
                                                   "settling velocity of tracers class"));

  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "tau_critical_erosion", offsetof(struct TracersContext_, tau_critical_erosion), 1,
                                                   "critical shear stress for erosion (N/m2)"));

  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "tau_critical_deposition", offsetof(struct TracersContext_, tau_critical_deposition),
                                                   1, "critical shear stress for deposition (N/m2)"));

  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "rhow", offsetof(struct TracersContext_, rhow), 1, "density of water"));

  PetscCallCEED(
      CeedQFunctionContextRegisterInt32(*qf_context, "tracers_ndof", offsetof(struct TracersContext_, tracers_ndof), 1, "number of tracers classes"));
  PetscCallCEED(CeedQFunctionContextRegisterInt32(*qf_context, "flow_ndof", offsetof(struct TracersContext_, flow_ndof), 1, "number of flow DoF"));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}
