// This code supports the C and Fortran MMS drivers and is not used in
// mainline RDycore (though it is built into the library).

#include <private/rdycoreimpl.h>

// this can be used in place of RDySetup for the MMS driver, which uses a
// modified YAML input schema (see ReadMMSConfigFile in yaml_input.c)
PetscErrorCode RDySetupMMS(RDy rdy) {
  PetscFunctionBegin;

  PetscCall(ReadMMSConfigFile(rdy));

  // override the log file name if necessary
  if (overridden_logfile_[0]) {
    strcpy(rdy->config.logging.file, overridden_logfile_);
  }

  // open the primary log file
  if (strlen(rdy->config.logging.file)) {
    PetscCall(PetscFOpen(rdy->comm, rdy->config.logging.file, "w", &rdy->log));
  } else {
    rdy->log = stdout;
  }

  // override parameters using command line arguments
  PetscCall(OverrideParameters(rdy));

  // initialize CEED if needed
  if (rdy->ceed_resource[0]) {
    PetscCallCEED(CeedInit(rdy->ceed_resource, &rdy->ceed));
  }

  RDyLogDebug(rdy, "Creating DMs...");
  PetscCall(CreateDM(rdy));           // for mesh and solution vector
  PetscCall(CreateAuxiliaryDM(rdy));  // for diagnostics
  PetscCall(CreateVectors(rdy));      // global and local vectors, residuals

  RDyLogDebug(rdy, "Initializing regions...");
  PetscCall(InitRegions(rdy));

  RDyLogDebug(rdy, "Initializing initial conditions and sources...");
  PetscCall(InitInitialConditions(rdy));
  PetscCall(InitSources(rdy));

  RDyLogDebug(rdy, "Creating solvers and vectors...");
  PetscCall(CreateSolvers(rdy));

  RDyLogDebug(rdy, "Creating FV mesh...");
  // note: this must be done after global vectors are created so a global
  // note: section exists for the DM
  PetscCall(RDyMeshCreateFromDM(rdy->dm, &rdy->mesh));

  RDyLogDebug(rdy, "Initializing boundaries and boundary conditions...");
  PetscCall(InitBoundaries(rdy));
  PetscCall(InitBoundaryConditions(rdy));

  RDyLogDebug(rdy, "Initializing solution data...");
  PetscCall(InitSolution(rdy));

  RDyLogDebug(rdy, "Initializing shallow water equations solver...");
  PetscCall(InitSWE(rdy));

  RDyLogDebug(rdy, "Initializing checkpoints...");
  PetscCall(InitCheckpoints(rdy));

  // if a restart has been requested, read the specified checkpoint file
  // and overwrite the necessary data
  if (rdy->config.restart.file[0]) {
    PetscCall(ReadCheckpointFile(rdy, rdy->config.restart.file));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
};
