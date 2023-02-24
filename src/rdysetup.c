#include <petscdmplex.h>
#include <private/rdycoreimpl.h>
#include <private/rdymemoryimpl.h>
#include <rdycore.h>

// sets default parameters
static PetscErrorCode SetDefaults(RDy rdy) {
  PetscFunctionBegin;

  rdy->config.log_level = LOG_INFO;

  // set the water depth below which no flow occurs
  rdy->config.tiny_h = 1e-7;

  PetscFunctionReturn(0);
}

// overrides parameters with command line arguments
static PetscErrorCode OverrideParameters(RDy rdy) {
  PetscFunctionBegin;

  if (rdy->dt <= 0.0) {
    // ѕet a default timestep if needed
    rdy->dt = rdy->config.final_time / rdy->config.max_step;
  }

  PetscOptionsBegin(rdy->comm, NULL, "RDycore options", "");
  { PetscCall(PetscOptionsReal("-dt", "dt", "", rdy->dt, &rdy->dt, NULL)); }
  PetscOptionsEnd();

  PetscFunctionReturn(0);
}

static PetscErrorCode CreateDM(RDy rdy) {
  PetscFunctionBegin;

  // read the grid from a file
  PetscCall(DMPlexCreateFromFile(rdy->comm, rdy->config.mesh_file, "grid", PETSC_TRUE, &rdy->dm));

  // interpolate the grid to get more connectivity
  {
    DM dm_interp;
    PetscCall(DMPlexInterpolate(rdy->dm, &dm_interp));
    PetscCheck(dm_interp, rdy->comm, PETSC_ERR_USER, "Mesh interpolation failed!");
    PetscCall(DMDestroy(&rdy->dm));
    rdy->dm = dm_interp;
  }

  // I'm not sure exactly why we need this here...
  PetscCall(DMPlexDistributeSetDefault(rdy->dm, PETSC_FALSE));

  // name the grid and apply any overrides from the command line
  PetscCall(PetscObjectSetName((PetscObject)rdy->dm, "grid"));
  PetscCall(DMSetFromOptions(rdy->dm));

  // create a section with (h, hu, hv) as degrees of freedom
  PetscInt     n_field            = 3;
  PetscInt     n_field_dof[3]     = {1, 1, 1};
  char         field_names[3][20] = {"height", "x momentum", "y momentum"};
  PetscSection sec;
  PetscCall(PetscSectionCreate(rdy->comm, &sec));
  PetscCall(PetscSectionSetNumFields(sec, n_field));
  PetscInt n_field_dof_tot = 0;
  for (PetscInt f = 0; f < n_field; ++f) {
    PetscCall(PetscSectionSetFieldName(sec, f, &field_names[f][0]));
    PetscCall(PetscSectionSetFieldComponents(sec, f, n_field_dof[f]));
    n_field_dof_tot += n_field_dof[f];
  }

  // set the number of degrees of freedom in each cell
  PetscInt c_start, c_end;  // starting and ending cell points
  PetscCall(DMPlexGetHeightStratum(rdy->dm, 0, &c_start, &c_end));
  PetscCall(PetscSectionSetChart(sec, c_start, c_end));
  for (PetscInt c = c_start; c < c_end; ++c) {
    for (PetscInt f = 0; f < n_field; ++f) {
      PetscCall(PetscSectionSetFieldDof(sec, c, f, n_field_dof[f]));
    }
    PetscCall(PetscSectionSetDof(sec, c, n_field_dof_tot));
  }

  // embed the section's data in our grid and toss the section
  PetscCall(PetscSectionSetUp(sec));
  PetscCall(DMSetLocalSection(rdy->dm, sec));
  PetscCall(PetscSectionDestroy(&sec));

  // set grid adacency and create a natural-to-local mapping
  PetscCall(DMSetBasicAdjacency(rdy->dm, PETSC_TRUE, PETSC_TRUE));
  PetscCall(DMSetUseNatural(rdy->dm, PETSC_TRUE));

  // distribute the mesh across processes
  {
    DM dm_dist;
    PetscCall(DMPlexDistribute(rdy->dm, 1, NULL, &dm_dist));
    if (dm_dist) {
      PetscCall(DMDestroy(&rdy->dm));
      rdy->dm = dm_dist;
    }
  }

  PetscCall(DMViewFromOptions(rdy->dm, NULL, "-dm_view"));

  PetscFunctionReturn(0);
}

// retrieves the index of a flow condition using its name
static PetscErrorCode FindFlowCondition(RDy rdy, const char *name, PetscInt *index) {
  PetscFunctionBegin;

  // Currently, we do a linear search on the name of the condition, which is O(N)
  // for N regions. If this is too slow, we can sort the conditions by name and
  // use binary search, which is O(log2 N).
  *index = -1;
  for (PetscInt i = 0; i < rdy->config.num_flow_conditions; ++i) {
    if (!strcmp(rdy->config.flow_conditions[i].name, name)) {
      *index = i;
      break;
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode CreateAuxiliaryDM(RDy rdy) {
  PetscFunctionBegin;

  PetscCall(DMClone(rdy->dm, &rdy->aux_dm));

  // create an auxiliary section with a diagnostic parameter.
  PetscInt     n_aux_field            = 1;
  PetscInt     n_aux_field_dof[1]     = {1};
  char         aux_field_names[1][20] = {"Parameter"};
  PetscSection aux_sec;
  PetscCall(PetscSectionCreate(rdy->comm, &aux_sec));
  PetscCall(PetscSectionSetNumFields(aux_sec, n_aux_field));
  PetscInt n_aux_field_dof_tot = 0;
  for (PetscInt f = 0; f < n_aux_field; ++f) {
    PetscCall(PetscSectionSetFieldName(aux_sec, f, &aux_field_names[f][0]));
    PetscCall(PetscSectionSetFieldComponents(aux_sec, f, n_aux_field_dof[f]));
    n_aux_field_dof_tot += n_aux_field_dof[f];
  }

  // set the number of auxiliary degrees of freedom in each cell
  PetscInt c_start, c_end;  // starting and ending cell points
  DMPlexGetHeightStratum(rdy->dm, 0, &c_start, &c_end);
  PetscCall(PetscSectionSetChart(aux_sec, c_start, c_end));
  for (PetscInt c = c_start; c < c_end; ++c) {
    for (PetscInt f = 0; f < n_aux_field; ++f) {
      PetscCall(PetscSectionSetFieldDof(aux_sec, c, f, n_aux_field_dof[f]));
    }
    PetscCall(PetscSectionSetDof(aux_sec, c, n_aux_field_dof_tot));
  }

  // embed the section's data in the auxiliary DM and toss the section
  PetscCall(PetscSectionSetUp(aux_sec));
  PetscCall(DMSetLocalSection(rdy->aux_dm, aux_sec));
  PetscCall(PetscSectionViewFromOptions(aux_sec, NULL, "-aux_layout_view"));
  PetscCall(PetscSectionDestroy(&aux_sec));

  // copy adjacency info from the primary DM
  PetscSF sf_migration, sf_natural;
  DMPlexGetMigrationSF(rdy->dm, &sf_migration);
  DMPlexCreateGlobalToNaturalSF(rdy->aux_dm, aux_sec, sf_migration, &sf_natural);
  DMPlexSetGlobalToNaturalSF(rdy->aux_dm, sf_natural);
  PetscSFDestroy(&sf_natural);

  PetscFunctionReturn(0);
}

// retrieves the index of a sediment condition using its name
static PetscErrorCode FindSedimentCondition(RDy rdy, const char *name, PetscInt *index) {
  PetscFunctionBegin;

  // Currently, we do a linear search on the name of the condition, which is O(N)
  // for N regions. If this is too slow, we can sort the conditions by name and
  // use binary search, which is O(log2 N).
  *index = -1;
  for (PetscInt i = 0; i < rdy->config.num_sediment_conditions; ++i) {
    if (!strcmp(rdy->config.sediment_conditions[i].name, name)) {
      *index = i;
      break;
    }
  }

  PetscFunctionReturn(0);
}

// retrieves the index of a salinity condition using its name
static PetscErrorCode FindSalinityCondition(RDy rdy, const char *name, PetscInt *index) {
  PetscFunctionBegin;

  // Currently, we do a linear search on the name of the condition, which is O(N)
  // for N regions. If this is too slow, we can sort the conditions by name and
  // use binary search, which is O(log2 N).
  *index = -1;
  for (PetscInt i = 0; i < rdy->config.num_salinity_conditions; ++i) {
    if (!strcmp(rdy->config.salinity_conditions[i].name, name)) {
      *index = i;
      break;
    }
  }

  PetscFunctionReturn(0);
}

// initializes mesh region/surface data
static PetscErrorCode InitRegionsAndSurfaces(RDy rdy) {
  PetscFunctionBegin;

  // Count and fetch regions.
  DMLabel label;
  PetscCall(DMGetLabel(rdy->dm, "Cell Sets", &label));
  for (PetscInt region_id = 0; region_id <= MAX_REGION_ID; ++region_id) {
    IS cell_is;
    PetscCall(DMLabelGetStratumIS(label, region_id, &cell_is));
    if (cell_is) ++rdy->num_regions;
    PetscCall(ISDestroy(&cell_is));
  }
  PetscCall(RDyAlloc(PetscInt, rdy->num_regions, &rdy->region_ids));
  PetscCall(RDyAlloc(RDyRegion, rdy->num_regions, &rdy->regions));
  PetscInt r = 0;
  for (PetscInt region_id = 0; region_id <= MAX_REGION_ID; ++region_id) {
    IS cell_is;  // cell index space
    PetscCall(DMLabelGetStratumIS(label, region_id, &cell_is));
    if (cell_is) {
      RDyRegion *region  = &rdy->regions[r];
      rdy->region_ids[r] = region_id;
      ++r;

      PetscInt num_cells;
      PetscCall(ISGetLocalSize(cell_is, &num_cells));
      if (num_cells > 0) {
        rdy->region_ids[rdy->num_regions] = region_id;
        region->num_cells                 = num_cells;
        PetscCall(RDyAlloc(PetscInt, region->num_cells, &region->cell_ids));
      }
      const PetscInt *cell_ids;
      PetscCall(ISGetIndices(cell_is, &cell_ids));
      memcpy(region->cell_ids, cell_ids, sizeof(PetscInt) * num_cells);
      PetscCall(ISRestoreIndices(cell_is, &cell_ids));
      PetscCall(ISDestroy(&cell_is));
    }
  }

  // Count and fetch surfaces.
  PetscCall(DMGetLabel(rdy->dm, "Face Sets", &label));
  for (PetscInt surface_id = 0; surface_id <= MAX_SURFACE_ID; ++surface_id) {
    IS edge_is;
    PetscCall(DMLabelGetStratumIS(label, surface_id, &edge_is));
    if (edge_is) ++rdy->num_surfaces;
    PetscCall(ISDestroy(&edge_is));
  }
  PetscCall(RDyAlloc(PetscInt, rdy->num_surfaces, &rdy->surface_ids));
  PetscCall(RDyAlloc(RDySurface, rdy->num_surfaces, &rdy->surfaces));
  PetscInt s = 0;
  for (PetscInt surface_id = 0; surface_id <= MAX_SURFACE_ID; ++surface_id) {
    IS edge_is;  // edge index space
    PetscCall(DMLabelGetStratumIS(label, surface_id, &edge_is));
    if (edge_is) {
      RDySurface *surface = &rdy->surfaces[s];
      rdy->surface_ids[s] = surface_id;
      ++s;

      PetscInt num_edges;
      PetscCall(ISGetLocalSize(edge_is, &num_edges));
      if (num_edges > 0) {
        surface->num_edges = num_edges;
        PetscCall(RDyAlloc(PetscInt, surface->num_edges, &surface->edge_ids));
      }
      const PetscInt *edge_ids;
      PetscCall(ISGetIndices(edge_is, &edge_ids));
      memcpy(surface->edge_ids, edge_ids, sizeof(PetscInt) * num_edges);
      PetscCall(ISRestoreIndices(edge_is, &edge_ids));
      PetscCall(ISDestroy(&edge_is));
    }
  }

  // make sure we have at least one region and surface
  PetscCheck(rdy->num_regions > 0, rdy->comm, PETSC_ERR_USER, "No regions were found in the grid!");
  PetscCheck(rdy->num_surfaces > 0, rdy->comm, PETSC_ERR_USER, "No surfaces were found in the grid!");

  PetscFunctionReturn(0);
}

// checks the validity of initial/boundary conditions and sources
static PetscErrorCode InitConditionsAndSources(RDy rdy) {
  PetscFunctionBegin;

  if (!strlen(rdy->config.initial_conditions_file)) {
    // Allocate storage for initial conditions.
    PetscCall(RDyAlloc(RDyCondition, rdy->num_regions, &rdy->initial_conditions));

    // Assign іnitial conditions to each region.
    for (PetscInt r = 0; r < rdy->num_regions; ++r) {
      RDyCondition     *ic        = &rdy->initial_conditions[r];
      PetscInt          region_id = rdy->region_ids[r];
      RDyConditionSpec *ic_spec   = &rdy->config.initial_conditions[region_id];

      PetscCheck(strlen(ic_spec->flow_name), rdy->comm, PETSC_ERR_USER, "Region %d has no initial flow condition!", region_id);
      PetscInt flow_index;
      PetscCall(FindFlowCondition(rdy, ic_spec->flow_name, &flow_index));
      RDyFlowCondition *flow_cond = &rdy->config.flow_conditions[flow_index];
      PetscCheck(flow_cond->type == CONDITION_DIRICHLET, rdy->comm, PETSC_ERR_USER,
                 "initial flow condition %s for region %d is not of dirichlet type!", flow_cond->name, region_id);
      ic->flow = flow_cond;

      if (rdy->config.sediment) {
        PetscCheck(strlen(ic_spec->sediment_name), rdy->comm, PETSC_ERR_USER, "Region %d has no initial sediment condition!", region_id);
        PetscInt sed_index;
        PetscCall(FindSedimentCondition(rdy, ic_spec->sediment_name, &sed_index));
        RDySedimentCondition *sed_cond = &rdy->config.sediment_conditions[sed_index];
        PetscCheck(sed_cond->type == CONDITION_DIRICHLET, rdy->comm, PETSC_ERR_USER,
                   "initial sediment condition %s for region %d is not of dirichlet type!", sed_cond->name, region_id);
        ic->sediment = sed_cond;
      }
      if (rdy->config.salinity) {
        PetscCheck(strlen(ic_spec->salinity_name), rdy->comm, PETSC_ERR_USER, "Region %d has no initial salinity condition!", region_id);
        PetscInt sal_index;
        PetscCall(FindSalinityCondition(rdy, ic_spec->salinity_name, &sal_index));
        RDySalinityCondition *sal_cond = &rdy->config.salinity_conditions[sal_index];
        PetscCheck(sal_cond->type == CONDITION_DIRICHLET, rdy->comm, PETSC_ERR_USER,
                   "initial salinity condition %s for region %d is not of dirichlet type!", sal_cond->name, region_id);
        ic->salinity = sal_cond;
      }
    }
  }
  if (rdy->config.num_sources > 0) {
    // Allocate storage for sources
    PetscCall(RDyAlloc(RDyCondition, rdy->num_regions, &rdy->sources));

    // Assign sources to each region as needed.
    for (PetscInt r = 0; r < rdy->num_regions; ++r) {
      RDyCondition     *src       = &rdy->sources[r];
      PetscInt          region_id = rdy->region_ids[r];
      RDyConditionSpec *src_spec  = &rdy->config.sources[region_id];
      if (strlen(src_spec->flow_name)) {
        PetscInt flow_index;
        PetscCall(FindFlowCondition(rdy, src_spec->flow_name, &flow_index));
        RDyFlowCondition *flow_cond = &rdy->config.flow_conditions[flow_index];
        PetscCheck(flow_cond->type == CONDITION_DIRICHLET, rdy->comm, PETSC_ERR_USER, "flow source %s for region %d is not of dirichlet type!",
                   flow_cond->name, region_id);
        src->flow = flow_cond;
      }

      if (rdy->config.sediment && strlen(src_spec->sediment_name)) {
        PetscInt sed_index;
        PetscCall(FindSedimentCondition(rdy, src_spec->sediment_name, &sed_index));
        RDySedimentCondition *sed_cond = &rdy->config.sediment_conditions[sed_index];
        PetscCheck(sed_cond->type == CONDITION_DIRICHLET, rdy->comm, PETSC_ERR_USER, "sediment source %s for region %d is not of dirichlet type!",
                   sed_cond->name, region_id);
        src->sediment = sed_cond;
      }
      if (rdy->config.salinity && strlen(src_spec->salinity_name)) {
        PetscInt sal_index;
        PetscCall(FindSalinityCondition(rdy, src_spec->salinity_name, &sal_index));
        RDySalinityCondition *sal_cond = &rdy->config.salinity_conditions[sal_index];
        PetscCheck(sal_cond->type == CONDITION_DIRICHLET, rdy->comm, PETSC_ERR_USER,
                   "initial salinity condition %s for region %d is not of dirichlet type!", sal_cond->name, region_id);
        src->salinity = sal_cond;
      }
    }
  }

  // Set up a reflecting flow boundary condition.
  RDyFlowCondition *reflecting_flow = NULL;
  for (PetscInt s = 0; s <= MAX_SURFACE_ID; ++s) {
    if (!strlen(rdy->config.flow_conditions[s].name)) {
      reflecting_flow = &rdy->config.flow_conditions[s];
      strcpy((char *)reflecting_flow->name, "reflecting");
      reflecting_flow->type = CONDITION_REFLECTING;
      break;
    }
  }
  PetscCheck(reflecting_flow, rdy->comm, PETSC_ERR_USER, "Could not allocate a reflecting flow condition! Please increase MAX_SURFACE_ID.");

  // Allocate storage for boundary conditions.
  PetscCall(RDyAlloc(RDyCondition, rdy->num_surfaces, &rdy->boundary_conditions));

  // Assign a boundary condition to each surface.
  for (PetscInt s = 0; s < rdy->num_surfaces; ++s) {
    RDyCondition     *bc         = &rdy->boundary_conditions[s];
    PetscInt          surface_id = rdy->surface_ids[s];
    RDyConditionSpec *bc_spec    = &rdy->config.boundary_conditions[surface_id];

    // If no flow condition was specified for a boundary, we set it to our
    // reflecting flow condition.
    if (!strlen(bc_spec->flow_name)) {
      bc->flow = reflecting_flow;
    } else {
      PetscInt flow_index;
      PetscCall(FindFlowCondition(rdy, bc_spec->flow_name, &flow_index));
      bc->flow = &rdy->config.flow_conditions[flow_index];
    }

    if (rdy->config.sediment) {
      PetscCheck(strlen(bc_spec->sediment_name), rdy->comm, PETSC_ERR_USER, "Surface %d has no sediment boundary condition!", surface_id);
      PetscInt sed_index;
      PetscCall(FindSedimentCondition(rdy, bc_spec->sediment_name, &sed_index));
      bc->sediment = &rdy->config.sediment_conditions[sed_index];
    }
    if (rdy->config.salinity) {
      PetscCheck(strlen(bc_spec->salinity_name), rdy->comm, PETSC_ERR_USER, "Surface %d has no salinity boundary condition!", surface_id);
      PetscInt sal_index;
      PetscCall(FindSalinityCondition(rdy, bc_spec->salinity_name, &sal_index));
      bc->salinity = &rdy->config.salinity_conditions[sal_index];
    }
  }

  PetscFunctionReturn(0);
}

// create solvers and vectors
static PetscErrorCode CreateSolvers(RDy rdy) {
  PetscFunctionBegin;

  // set up vectors
  PetscCall(DMCreateGlobalVector(rdy->dm, &rdy->X));
  PetscCall(VecDuplicate(rdy->X, &rdy->R));
  PetscCall(VecViewFromOptions(rdy->X, NULL, "-vec_view"));
  PetscCall(DMCreateLocalVector(rdy->dm, &rdy->X_local));

  PetscInt n_dof;
  PetscCall(VecGetSize(rdy->X, &n_dof));

  // set up a TS solver
  PetscCall(TSCreate(rdy->comm, &rdy->ts));
  PetscCall(TSSetProblemType(rdy->ts, TS_NONLINEAR));
  switch (rdy->config.temporal) {
    case TEMPORAL_EULER:
      PetscCall(TSSetType(rdy->ts, TSEULER));
      break;
    case TEMPORAL_RK4:
      PetscCall(TSSetType(rdy->ts, TSRK));
      PetscCall(TSRKSetType(rdy->ts, TSRK4));
      break;
    case TEMPORAL_BEULER:
      PetscCall(TSSetType(rdy->ts, TSBEULER));
      break;
  }
  PetscCall(TSSetDM(rdy->ts, rdy->dm));

  PetscCheck(rdy->config.flow_mode == FLOW_SWE, rdy->comm, PETSC_ERR_USER, "Only the 'swe' flow mode is currently supported.");
  PetscCall(InitSWE(rdy));  // initialize SWE physics
  PetscCall(TSSetRHSFunction(rdy->ts, rdy->R, RHSFunctionSWE, rdy));

  PetscCall(TSSetMaxTime(rdy->ts, rdy->config.final_time));
  PetscCall(TSSetExactFinalTime(rdy->ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetSolution(rdy->ts, rdy->X));
  PetscCall(TSSetTimeStep(rdy->ts, rdy->dt));

  // apply any solver-related options supplied on the command line
  PetscCall(TSSetFromOptions(rdy->ts));
  PetscCall(TSGetTimeStep(rdy->ts, &rdy->dt));  // just in case!

  PetscFunctionReturn(0);
}

// initializes solution vector data
static PetscErrorCode InitSolution(RDy rdy) {
  PetscFunctionBegin;

  PetscCall(VecZeroEntries(rdy->X));
  if (strlen(rdy->config.initial_conditions_file)) {  // read from file
    PetscViewer viewer;
    PetscCall(PetscViewerBinaryOpen(rdy->comm, rdy->config.initial_conditions_file, FILE_MODE_READ, &viewer));
    Vec natural;
    PetscCall(DMPlexCreateNaturalVector(rdy->dm, &natural));
    PetscCall(VecLoad(natural, viewer));
    PetscCall(DMPlexNaturalToGlobalBegin(rdy->dm, natural, rdy->X));
    PetscCall(DMPlexNaturalToGlobalEnd(rdy->dm, natural, rdy->X));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(VecDestroy(&natural));
  } else {
    // we initialize from specified initial conditions by looping over regions
    // and writing values for corresponding cells
    PetscInt n_local;
    PetscCall(VecGetLocalSize(rdy->X, &n_local));
    PetscScalar *x_ptr;
    PetscCall(VecGetArray(rdy->X, &x_ptr));
    for (PetscInt r = 0; r < rdy->num_regions; ++r) {
      RDyRegion    *region = &rdy->regions[r];
      RDyCondition *ic     = &rdy->initial_conditions[r];
      for (PetscInt c = 0; c < region->num_cells; ++c) {
        PetscInt cell_id = region->cell_ids[c];
        if (3 * cell_id < n_local) {  // skip ghost cells
          x_ptr[3 * cell_id]     = ic->flow->height;
          x_ptr[3 * cell_id + 1] = ic->flow->momentum[0];
          x_ptr[3 * cell_id + 2] = ic->flow->momentum[1];
        }
      }
    }
    PetscCall(VecRestoreArray(rdy->X, &x_ptr));
  }

  PetscFunctionReturn(0);
}

/// Performs any setup needed by RDy, reading from the specified configuration
/// file.
PetscErrorCode RDySetup(RDy rdy) {
  PetscFunctionBegin;

  PetscCall(SetDefaults(rdy));
  PetscCall(ReadConfigFile(rdy));

  // open the primary log file
  if (strlen(rdy->config.log_file)) {
    PetscCall(PetscFOpen(rdy->comm, rdy->config.log_file, "w", &rdy->log));
  } else {
    rdy->log = stdout;
  }

  // override parameters using command line arguments
  PetscCall(OverrideParameters(rdy));

  // print configuration info
  PetscCall(PrintConfig(rdy));

  RDyLogDebug(rdy, "Creating DMs...");
  PetscCall(CreateDM(rdy));           // for mesh and solution vector
  PetscCall(CreateAuxiliaryDM(rdy));  // for diagnostics

  RDyLogDebug(rdy, "Initializing regions and surfaces...");
  PetscCall(InitRegionsAndSurfaces(rdy));

  RDyLogDebug(rdy, "Initializing initial/boundary conditions and sources...");
  PetscCall(InitConditionsAndSources(rdy));

  RDyLogDebug(rdy, "Creating solvers and vectors...");
  PetscCall(CreateSolvers(rdy));

  RDyLogDebug(rdy, "Creating FV mesh...");
  // note: this must be done after global vectors are created so a global
  // note: section exists for the DM
  PetscCall(RDyMeshCreateFromDM(rdy->dm, &rdy->mesh));

  RDyLogDebug(rdy, "Initializing solution data...");
  PetscCall(InitSolution(rdy));

  PetscFunctionReturn(0);
}
