// This code supports the C and Fortran MMS drivers and is not used in
// mainline RDycore (though it is built into the library).

#include <muParserDLL.h>
#include <petscdmceed.h>
#include <petscdmplex.h>
#include <petscsys.h>
#include <private/rdycoreimpl.h>
#include <private/rdydmimpl.h>

static PetscErrorCode SetAnalyticInitialCondition(RDy rdy) {
  PetscFunctionBegin;

  // we have a single analytical solution on the entire domain, and we use
  // that solution as an initial condition

  // allocate storage for by-region initial conditions
  PetscCall(PetscCalloc1(rdy->num_regions, &rdy->initial_conditions));

  for (PetscInt r = 0; r < rdy->num_regions; ++r) {
    RDyCondition *ic              = &rdy->initial_conditions[r];
    RDyRegion     region          = rdy->regions[r];
    PetscInt      region_ic_index = -1;
    for (PetscInt ic = 0; ic < rdy->config.num_initial_conditions; ++ic) {
      if (!strcmp(rdy->config.initial_conditions[ic].region, region.name)) {
        region_ic_index = ic;
        break;
      }
    }
    PetscCheck(region_ic_index != -1, rdy->comm, PETSC_ERR_USER, "Region '%s' has no initial conditions!", region.name);

    RDyRegionConditionSpec ic_spec = rdy->config.initial_conditions[region_ic_index];
    PetscCheck(strlen(ic_spec.flow), rdy->comm, PETSC_ERR_USER, "Region '%s' has no initial flow condition!", region.name);
    PetscInt flow_index;
    PetscCall(FindFlowCondition(rdy, ic_spec.flow, &flow_index));
    PetscCheck(flow_index != -1, rdy->comm, PETSC_ERR_USER, "initial flow condition '%s' for region '%s' was not found!", ic_spec.flow, region.name);
    RDyFlowCondition *flow_cond = &rdy->config.flow_conditions[flow_index];
    PetscCheck(flow_cond->type == CONDITION_DIRICHLET, rdy->comm, PETSC_ERR_USER,
               "initial flow condition '%s' for region '%s' is not of dirichlet type!", flow_cond->name, region.name);
    ic->flow = flow_cond;

    if (rdy->config.physics.sediment) {
      PetscCheck(strlen(ic_spec.sediment), rdy->comm, PETSC_ERR_USER, "Region '%s' has no initial sediment condition!", region.name);
      PetscInt sed_index;
      PetscCall(FindSedimentCondition(rdy, ic_spec.sediment, &sed_index));
      RDySedimentCondition *sed_cond = &rdy->config.sediment_conditions[sed_index];
      PetscCheck(sed_cond->type == CONDITION_DIRICHLET, rdy->comm, PETSC_ERR_USER,
                 "initial sediment condition '%s' for region '%s' is not of dirichlet type!", sed_cond->name, region.name);
      ic->sediment = sed_cond;
    }
    if (rdy->config.physics.salinity) {
      PetscCheck(strlen(ic_spec.salinity), rdy->comm, PETSC_ERR_USER, "Region '%s' has no initial salinity condition!", region.name);
      PetscInt sal_index;
      PetscCall(FindSalinityCondition(rdy, ic_spec.salinity, &sal_index));
      RDySalinityCondition *sal_cond = &rdy->config.salinity_conditions[sal_index];
      PetscCheck(sal_cond->type == CONDITION_DIRICHLET, rdy->comm, PETSC_ERR_USER,
                 "initial salinity condition '%s' for region '%s' is not of dirichlet type!", sal_cond->name, region.name);
      ic->salinity = sal_cond;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetAnalyticSource(RDy rdy) {
  PetscFunctionBegin;
  if (rdy->config.num_sources > 0) {
    // allocate storage for sources
    PetscCall(PetscCalloc1(rdy->num_regions, &rdy->sources));

    // assign sources to each region as needed
    for (PetscInt r = 0; r < rdy->num_regions; ++r) {
      RDyCondition *src              = &rdy->sources[r];
      RDyRegion     region           = rdy->regions[r];
      PetscInt      region_src_index = -1;
      for (PetscInt isrc = 0; isrc < rdy->config.num_sources; ++isrc) {
        if (!strcmp(rdy->config.sources[isrc].region, region.name)) {
          region_src_index = isrc;
          break;
        }
      }
      if (region_src_index != -1) {
        RDyRegionConditionSpec src_spec = rdy->config.sources[region_src_index];
        if (strlen(src_spec.flow)) {
          PetscInt flow_index;
          PetscCall(FindFlowCondition(rdy, src_spec.flow, &flow_index));
          PetscCheck(flow_index != -1, rdy->comm, PETSC_ERR_USER, "source flow condition '%s' for region '%s' was not found!", src_spec.flow,
                     region.name);
          RDyFlowCondition *flow_cond = &rdy->config.flow_conditions[flow_index];
          PetscCheck(flow_cond->type == CONDITION_DIRICHLET, rdy->comm, PETSC_ERR_USER, "flow source '%s' for region '%s' is not of dirichlet type!",
                     flow_cond->name, region.name);
          src->flow = flow_cond;
        }

        if (rdy->config.physics.sediment && strlen(src_spec.sediment)) {
          PetscInt sed_index;
          PetscCall(FindSedimentCondition(rdy, src_spec.sediment, &sed_index));
          RDySedimentCondition *sed_cond = &rdy->config.sediment_conditions[sed_index];
          PetscCheck(sed_cond->type == CONDITION_DIRICHLET, rdy->comm, PETSC_ERR_USER,
                     "sediment source '%s' for region '%s' is not of dirichlet type!", sed_cond->name, region.name);
          src->sediment = sed_cond;
        }
        if (rdy->config.physics.salinity && strlen(src_spec.salinity)) {
          PetscInt sal_index;
          PetscCall(FindSalinityCondition(rdy, src_spec.salinity, &sal_index));
          RDySalinityCondition *sal_cond = &rdy->config.salinity_conditions[sal_index];
          PetscCheck(sal_cond->type == CONDITION_DIRICHLET, rdy->comm, PETSC_ERR_USER,
                     "initial salinity condition '%s' for region '%s' is not of dirichlet type!", sal_cond->name, region.name);
          src->salinity = sal_cond;
        }
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetAnalyticBoundaryCondition(RDy rdy) {
  PetscFunctionBegin;
  // Set up a reflecting flow boundary condition.
  RDyFlowCondition *reflecting_flow = NULL;
  for (PetscInt c = 0; c < MAX_NUM_CONDITIONS; ++c) {
    if (!strlen(rdy->config.flow_conditions[c].name)) {
      reflecting_flow = &rdy->config.flow_conditions[c];
      strcpy((char *)reflecting_flow->name, "reflecting");
      reflecting_flow->type = CONDITION_REFLECTING;
      break;
    }
  }
  PetscCheck(reflecting_flow, rdy->comm, PETSC_ERR_USER, "Could not allocate a reflecting flow condition! Please increase MAX_BOUNDARY_ID.");

  // Allocate storage for boundary conditions.
  PetscCall(PetscCalloc1(rdy->num_boundaries, &rdy->boundary_conditions));

  // Assign a boundary condition to each boundary.
  for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
    RDyCondition *bc       = &rdy->boundary_conditions[b];
    RDyBoundary   boundary = rdy->boundaries[b];

    // identify the index of the boundary condition assigned to this boundary.
    PetscInt bc_index = -1;
    for (PetscInt ib = 0; ib < rdy->config.num_boundary_conditions; ++ib) {
      RDyBoundaryConditionSpec bc_spec = rdy->config.boundary_conditions[ib];
      for (PetscInt ib1 = 0; ib1 < bc_spec.num_boundaries; ++ib1) {
        if (!strcmp(bc_spec.boundaries[ib1], boundary.name)) {
          PetscCheck(bc_index == -1, rdy->comm, PETSC_ERR_USER, "Boundary '%s' is assigned to more than one boundary condition!", boundary.name);
          bc_index = ib;
          break;
        }
      }
    }
    if (bc_index != -1) {
      RDyBoundaryConditionSpec bc_spec = rdy->config.boundary_conditions[bc_index];

      // If no flow condition was specified for a boundary, we set it to our
      // reflecting flow condition.
      if (!strlen(bc_spec.flow)) {
        bc->flow = reflecting_flow;
      } else {
        PetscInt flow_index;
        PetscCall(FindFlowCondition(rdy, bc_spec.flow, &flow_index));
        PetscCheck(flow_index != -1, rdy->comm, PETSC_ERR_USER, "boundary flow condition '%s' for boundary '%s' was not found!", bc_spec.flow,
                   boundary.name);
        bc->flow = &rdy->config.flow_conditions[flow_index];
      }

      if (rdy->config.physics.sediment) {
        PetscCheck(strlen(bc_spec.sediment), rdy->comm, PETSC_ERR_USER, "Boundary '%s' has no sediment boundary condition!", boundary.name);
        PetscInt sed_index;
        PetscCall(FindSedimentCondition(rdy, bc_spec.sediment, &sed_index));
        bc->sediment = &rdy->config.sediment_conditions[sed_index];
      }
      if (rdy->config.physics.salinity) {
        PetscCheck(strlen(bc_spec.salinity), rdy->comm, PETSC_ERR_USER, "Boundary '%s' has no salinity boundary condition!", boundary.name);
        PetscInt sal_index;
        PetscCall(FindSalinityCondition(rdy, bc_spec.salinity, &sal_index));
        bc->salinity = &rdy->config.salinity_conditions[sal_index];
      }
    } else {
      // this boundary wasn't explicitly requested, so set up an auto-generated
      // reflecting BC
      bc->flow           = reflecting_flow;
      bc->auto_generated = PETSC_TRUE;
    }
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

  RDyLogDebug(rdy, "Initializing initial conditions and sources...");
  PetscCall(SetAnalyticInitialCondition(rdy));
  PetscCall(SetAnalyticSource(rdy));

  RDyLogDebug(rdy, "Creating FV mesh...");
  // note: this must be done after global vectors are created so a global
  // note: section exists for the DM
  PetscCall(RDyMeshCreateFromDM(rdy->dm, &rdy->mesh));

  RDyLogDebug(rdy, "Initializing boundaries and boundary conditions...");
  PetscCall(InitBoundaries(rdy));
  PetscCall(SetAnalyticBoundaryCondition(rdy));

  RDyLogDebug(rdy, "Initializing solution data...");
  PetscCall(SetAnalyticSolution(rdy));

  RDyLogDebug(rdy, "Initializing shallow water equations solver...");
  PetscCall(InitSWE(rdy));

  PetscFunctionReturn(PETSC_SUCCESS);
};
