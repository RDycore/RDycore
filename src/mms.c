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
  PetscCall(InitMMSInitialConditions(rdy));
  PetscCall(InitMMSSources(rdy));

  RDyLogDebug(rdy, "Creating solvers and vectors...");
  PetscCall(CreateSolvers(rdy));

  RDyLogDebug(rdy, "Creating FV mesh...");
  // note: this must be done after global vectors are created so a global
  // note: section exists for the DM
  PetscCall(RDyMeshCreateFromDM(rdy->dm, &rdy->mesh));

  RDyLogDebug(rdy, "Initializing boundaries and boundary conditions...");
  PetscCall(InitBoundaries(rdy));
  PetscCall(InitMMSBoundaryConditions(rdy));

  RDyLogDebug(rdy, "Initializing solution data...");
  PetscCall(InitMMSSolution(rdy));

  PetscCall(CreateSWEOperators(rdy));
  if (rdy->ceed_resource[0]) {
    RDyLogDebug(rdy, "Setting up CEED Operators...");

    // create the operators themselves
    PetscCall(CreateSWEFluxOperator(rdy->ceed, &rdy->mesh, rdy->num_boundaries, rdy->boundaries, rdy->boundary_conditions,
                                    rdy->config.physics.flow.tiny_h, &rdy->ceed_rhs.op_edges));

    PetscCall(CreateSWESourceOperator(rdy->ceed, &rdy->mesh, rdy->mesh.num_cells, rdy->materials_by_cell, rdy->config.physics.flow.tiny_h,
                                      &rdy->ceed_rhs.op_src));

    // create associated vectors for storage
    int num_comp = 3;
    PetscCallCEED(CeedVectorCreate(rdy->ceed, rdy->mesh.num_cells * num_comp, &rdy->ceed_rhs.u_local_ceed));
    PetscCallCEED(CeedVectorCreate(rdy->ceed, rdy->mesh.num_cells * num_comp, &rdy->ceed_rhs.f_ceed));
    PetscCallCEED(CeedVectorCreate(rdy->ceed, rdy->mesh.num_cells_local * num_comp, &rdy->ceed_rhs.s_ceed));
    PetscCallCEED(CeedVectorCreate(rdy->ceed, rdy->mesh.num_cells_local * num_comp, &rdy->ceed_rhs.u_ceed));

    // reset the time step size
    rdy->ceed_rhs.dt = 0.0;
  } else {
    // allocate storage for our PETSc implementation of the  flux and
    // source terms
    RDyLogDebug(rdy, "Allocating PETSc data structures for fluxes and sources...");
    PetscCall(CreatePetscSWEFlux(rdy->mesh.num_internal_edges, rdy->num_boundaries, rdy->boundaries, &rdy->petsc_rhs));
    PetscCall(CreatePetscSWESource(&rdy->mesh, rdy->petsc_rhs));
  }

  // make sure any Dirichlet boundary conditions are properly specified
  PetscCall(InitDirichletBoundaryConditions(rdy));

  RDyLogDebug(rdy, "Initializing checkpoints...");
  PetscCall(InitCheckpoints(rdy));

  // if a restart has been requested, read the specified checkpoint file
  // and overwrite the necessary data
  if (rdy->config.restart.file[0]) {
    PetscCall(ReadCheckpointFile(rdy, rdy->config.restart.file));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
};
