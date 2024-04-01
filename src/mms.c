// This code supports the C and Fortran MMS drivers and is not used in
// mainline RDycore (though it is built into the library).

#include <muParserDLL.h>
#include <petscdmceed.h>
#include <petscdmplex.h>
#include <petscsys.h>
#include <private/rdycoreimpl.h>
#include <private/rdydmimpl.h>

static PetscErrorCode SetAnalyticSource(RDy rdy) {
  PetscFunctionBegin;
  if (rdy->config.num_sources > 0) {
    // We only need a single Dirichlet boundary condition whose data can be
    // set to the analytic solution as needed.
    RDyCondition analytic_source = {};

    // allocate storage for sources
    PetscCall(PetscCalloc1(rdy->num_regions, &rdy->sources));

    // assign all regions to the analytic source
    for (PetscInt r = 0; r < rdy->num_regions; ++r) {
      rdy->sources[r] = analytic_source;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetAnalyticBoundaryCondition(RDy rdy) {
  PetscFunctionBegin;

  // We only need a single Dirichlet boundary condition, populated with
  // manufactured solution data.

  RDyCondition analytic_bc = {};

  // Assign the boundary condition to each boundary.
  PetscCall(PetscCalloc1(rdy->num_boundaries, &rdy->boundary_conditions));
  for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
    rdy->boundary_conditions[b] = analytic_bc;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetAnalyticSolution(RDy rdy) {
  PetscFunctionBegin;

  PetscCall(VecZeroEntries(rdy->X));

  // check that each region has an initial condition
  for (PetscInt r = 0; r < rdy->num_regions; ++r) {
    RDyRegion    region = rdy->regions[r];
    RDyCondition ic     = rdy->initial_conditions[r];
    PetscCheck(ic.flow, rdy->comm, PETSC_ERR_USER, "No initial condition specified for region '%s'", region.name);
  }

  // now initialize or override initial conditions for each region
  PetscInt n_local, ndof;
  PetscCall(VecGetLocalSize(rdy->X, &n_local));
  PetscCall(VecGetBlockSize(rdy->X, &ndof));
  PetscScalar *x_ptr;
  PetscCall(VecGetArray(rdy->X, &x_ptr));

  // initialize flow conditions
  for (PetscInt f = 0; f < rdy->config.num_flow_conditions; ++f) {
    RDyFlowCondition flow_ic = rdy->config.flow_conditions[f];
    Vec              local   = NULL;
    if (flow_ic.file[0]) {  // read flow data from file
      PetscViewer viewer;
      PetscCall(PetscViewerBinaryOpen(rdy->comm, flow_ic.file, FILE_MODE_READ, &viewer));

      Vec natural, global;
      PetscCall(DMPlexCreateNaturalVector(rdy->dm, &natural));
      PetscCall(DMCreateGlobalVector(rdy->dm, &global));
      PetscCall(DMCreateLocalVector(rdy->dm, &local));

      PetscCall(VecLoad(natural, viewer));
      PetscCall(PetscViewerDestroy(&viewer));

      // check the block size of the initial condition vector agrees with the block size of rdy->X
      PetscInt nblocks_nat;
      PetscCall(VecGetBlockSize(natural, &nblocks_nat));
      PetscCheck((ndof == nblocks_nat), rdy->comm, PETSC_ERR_USER,
                 "The block size of the initial condition ('%" PetscInt_FMT
                 "') "
                 "does not match with the number of DOFs ('%" PetscInt_FMT "')",
                 nblocks_nat, ndof);

      // scatter natural-to-global
      PetscCall(DMPlexNaturalToGlobalBegin(rdy->dm, natural, global));
      PetscCall(DMPlexNaturalToGlobalEnd(rdy->dm, natural, global));

      // scatter global-to-local
      PetscCall(DMGlobalToLocalBegin(rdy->dm, global, INSERT_VALUES, local));
      PetscCall(DMGlobalToLocalEnd(rdy->dm, global, INSERT_VALUES, local));

      // free up memory
      PetscCall(VecDestroy(&natural));
      PetscCall(VecDestroy(&global));
    }

    // set regional flow as needed
    for (PetscInt r = 0; r < rdy->num_regions; ++r) {
      RDyRegion    region = rdy->regions[r];
      RDyCondition ic     = rdy->initial_conditions[r];
      if (!strcmp(ic.flow->name, flow_ic.name)) {
        if (local) {
          PetscScalar *local_ptr;
          PetscCall(VecGetArray(local, &local_ptr));
          for (PetscInt c = 0; c < region.num_cells; ++c) {
            PetscInt cell_id = region.cell_ids[c];
            if (ndof * cell_id < n_local) {  // skip ghost cells
              for (PetscInt idof = 0; idof < ndof; idof++) {
                x_ptr[ndof * cell_id + idof] = local_ptr[ndof * cell_id + idof];
              }
            }
          }
          PetscCall(VecRestoreArray(local, &local_ptr));
        } else {
          for (PetscInt c = 0; c < region.num_cells; ++c) {
            PetscInt cell_id = region.cell_ids[c];
            if (ndof * cell_id < n_local) {  // skip ghost cells
              x_ptr[3 * cell_id]     = mupEval(flow_ic.height);
              x_ptr[3 * cell_id + 1] = mupEval(flow_ic.momentum[0]);
              x_ptr[3 * cell_id + 2] = mupEval(flow_ic.momentum[1]);
            }
          }
        }
      }
    }
    PetscCall(VecDestroy(&local));
  }

  // TODO: salinity and sediment initial conditions go here.

  PetscCall(VecRestoreArray(rdy->X, &x_ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// this can be used in place of RDySetup for the MMS driver, which uses a
// modified YAML input schema (see ReadMMSConfigFile in yaml_input.c)
PetscErrorCode RDySetupMMS(RDy rdy) {
  PetscFunctionBegin;

  PetscCall(ReadMMSConfigFile(rdy));

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

  RDyLogDebug(rdy, "Creating FV mesh...");
  // note: this must be done after global vectors are created so a global
  // note: section exists for the DM
  PetscCall(RDyMeshCreateFromDM(rdy->dm, &rdy->mesh));

  RDyLogDebug(rdy, "Initializing boundaries and boundary conditions...");
  PetscCall(InitBoundaries(rdy));
  PetscCall(SetAnalyticBoundaryCondition(rdy));

  RDyLogDebug(rdy, "Initializing solution and source data...");
  PetscCall(SetAnalyticSolution(rdy));
  PetscCall(SetAnalyticSource(rdy));

  RDyLogDebug(rdy, "Initializing shallow water equations solver...");
  PetscCall(InitSWE(rdy));

  PetscFunctionReturn(PETSC_SUCCESS);
};
