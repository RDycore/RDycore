#include <ceed/ceed.h>
#include <petscdmceed.h>
#include <private/rdysweimpl.h>

// these are used to register CEED operator events for profiling
static PetscClassId RDY_CLASSID;
PetscLogEvent       RDY_CeedOperatorApply;

/// Initializes solvers for physics specified in the input configuration.
PetscErrorCode InitSolvers(RDy rdy) {
  PetscFunctionBegin;

  // register a logging event for applying our CEED operator
  PetscCall(PetscClassIdRegister("RDycore", &RDY_CLASSID));
  PetscCall(PetscLogEventRegister("CeedOperatorApp", RDY_CLASSID, &RDY_CeedOperatorApply));

  // just pass the call along for now
  InitSWE(rdy);

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Frees all resources devoted to solvers.
PetscErrorCode DestroySolvers(RDy rdy) {
  PetscFunctionBegin;

  PetscBool ceed_enabled = CeedEnabled(rdy);

  if (rdy->petsc.context) {
    PetscCall(DestroyPetscSWEFlux(rdy->petsc.context, ceed_enabled, rdy->num_boundaries));
  }

  if (ceed_enabled) {
    PetscCallCEED(CeedOperatorDestroy(&rdy->ceed.flux_operator));
    PetscCallCEED(CeedOperatorDestroy(&rdy->ceed.source_operator));
    PetscCallCEED(CeedVectorDestroy(&rdy->ceed.u_local));
    PetscCallCEED(CeedVectorDestroy(&rdy->ceed.rhs));
    PetscCallCEED(CeedVectorDestroy(&rdy->ceed.sources));
    if (rdy->ceed.host_fluxes) VecDestroy(&rdy->ceed.host_fluxes);
    // the CEED context belongs to RDycore itself, so we don't delete it
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
