#include <petscdmplex.h>
#include <private/rdycoreimpl.h>
#include <private/rdymemoryimpl.h>
#include <rdycore.h>

extern PetscErrorCode ReadConfigFile(RDy rdy);
extern PetscErrorCode PrintConfig(RDy rdy);

// sets default parameters
static PetscErrorCode SetDefaults(RDy rdy) {
  PetscFunctionBegin;

  rdy->config.log_level = LOG_INFO;

  PetscFunctionReturn(0);
}

// overrides parameters with command line arguments
static PetscErrorCode OverrideParameters(RDy rdy) {
  PetscFunctionBegin;

  // FIXME

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
        ++rdy->num_regions;
        region->num_cells = num_cells;
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
      RDyLogDebug(rdy, "Setting reflecting flow condition for surface %d\n", surface_id);
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

// Configures bookkeeping data structures for the simulation
static PetscErrorCode InitSimulationData(RDy rdy) {
  PetscFunctionBegin;

  // FIXME

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

  // create the grid from the given file and distribute it amongst processes.
  PetscCall(DMPlexCreateFromFile(rdy->comm, rdy->config.mesh_file, "grid", PETSC_TRUE, &rdy->dm));
  DM dm_dist;
  PetscCall(DMPlexDistribute(rdy->dm, 1, NULL, &dm_dist));
  if (dm_dist) {
    PetscCall(DMDestroy(&rdy->dm));
    rdy->dm = dm_dist;
  }

  // set up mesh regions and surfaces, reading them from our DMPlex object
  PetscCall(InitRegionsAndSurfaces(rdy));

  // set up initial/boundary conditions and sources
  PetscCall(InitConditionsAndSources(rdy));

  // set up bookkeeping data structures
  PetscCall(InitSimulationData(rdy));

  PetscFunctionReturn(0);
}
