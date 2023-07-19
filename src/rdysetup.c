#include <petscdmplex.h>
#include <private/rdycoreimpl.h>
#include <private/rdymemoryimpl.h>
#include <rdycore.h>

// Maximum length of the name of a prognostic or diagnostic field component
#define MAX_COMP_NAME_LENGTH 20

// time conversion factors
static const PetscReal secs_in_min = 60.0;
static const PetscReal mins_in_hr  = 60.0;
static const PetscReal hrs_in_day  = 24.0;
static const PetscReal days_in_mon = 30.0;
static const PetscReal days_in_yr  = 365.0;

/// Returns a string corresponding to the given time unit
const char *TimeUnitAsString(RDyTimeUnit time_unit) {
  static const char *time_unit_strings[6] = {
      "sec",  // seconds
      "min",  // minutes
      "hr",   // hours
      "day",  // days
      "mon",  // months
      "yr",   // years
  };
  PetscFunctionBegin;
  PetscFunctionReturn(time_unit_strings[time_unit]);
}

/// Converts the given time (expressed in the given units) to seconds.
/// @param [in] time the time as expressed in the given units
/// @param [in] time_unit the units in which the time is expressed
PetscReal ConvertTimeToSeconds(PetscReal time, RDyTimeUnit time_unit) {
  PetscFunctionBegin;

  PetscReal time_in_sec;
  switch (time_unit) {
    case TIME_SECONDS:
      time_in_sec = time;
      break;
    case TIME_MINUTES:
      time_in_sec = time * secs_in_min;
      break;
    case TIME_HOURS:
      time_in_sec = time * mins_in_hr * secs_in_min;
      break;
    case TIME_DAYS:
      time_in_sec = time * hrs_in_day * mins_in_hr * secs_in_min;
      break;
    case TIME_MONTHS:
      time_in_sec = time * days_in_mon * hrs_in_day * mins_in_hr * secs_in_min;
      break;
    case TIME_YEARS:
      time_in_sec = time * days_in_yr * hrs_in_day * mins_in_hr * secs_in_min;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Unsupported time unit");
      break;
  }

  PetscFunctionReturn(time_in_sec);
}

/// Converts the given time (expressed in seconds) to the given units.
/// @param [in] time the time as expressed in seconds
/// @param [in] time_unit the units to which the time is to be converted
PetscReal ConvertTimeFromSeconds(PetscReal time, RDyTimeUnit time_unit) {
  PetscFunctionBegin;
  PetscReal time_in_units;
  switch (time_unit) {
    case TIME_SECONDS:
      time_in_units = time;
      break;
    case TIME_MINUTES:
      time_in_units = time / secs_in_min;
      break;
    case TIME_HOURS:
      time_in_units = time / mins_in_hr / secs_in_min;
      break;
    case TIME_DAYS:
      time_in_units = time / hrs_in_day / mins_in_hr / secs_in_min;
      break;
    case TIME_MONTHS:
      time_in_units = time / days_in_mon / hrs_in_day / mins_in_hr / secs_in_min;
      break;
    case TIME_YEARS:
      time_in_units = time / days_in_yr / hrs_in_day / mins_in_hr / secs_in_min;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Unsupported time unit");
      break;
  }

  PetscFunctionReturn(time_in_units);
}

// overrides parameters with command line arguments
static PetscErrorCode OverrideParameters(RDy rdy) {
  PetscFunctionBegin;

  if (rdy->dt <= 0.0) {
    // ѕet a default timestep if needed
    rdy->dt = ConvertTimeToSeconds(rdy->config.time.final_time, rdy->config.time.unit) / rdy->config.time.max_step;
  } else {
    // convert dt to seconds in any case
    rdy->dt = ConvertTimeToSeconds(rdy->dt, rdy->config.time.unit);
  }

  PetscOptionsBegin(rdy->comm, NULL, "RDycore options", "");
  {
    PetscCall(PetscOptionsReal("-dt", "dt (seconds)", "", rdy->dt, &rdy->dt, NULL));
    PetscCall(PetscOptionsString("-ceed", "Ceed resource (/cpu/self, /gpu/cuda, /gpu/hip, ...)", "", rdy->ceed_resource, rdy->ceed_resource,
                                 sizeof rdy->ceed_resource, NULL));
  }
  PetscOptionsEnd();

  PetscFunctionReturn(0);
}

static PetscErrorCode CreateDM(RDy rdy) {
  PetscFunctionBegin;

  // read the grid from a file
  PetscCall(DMPlexCreateFromFile(rdy->comm, rdy->config.grid.file, "grid", PETSC_TRUE, &rdy->dm));

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
  PetscInt n_field                             = 1;
  PetscInt n_field_comps[1]                    = {3};
  char     comp_names[3][MAX_COMP_NAME_LENGTH] = {
          "Height",
          "MomentumX",
          "MomentumY",
  };
  PetscSection sec;
  PetscCall(PetscSectionCreate(rdy->comm, &sec));
  PetscCall(PetscSectionSetNumFields(sec, n_field));
  PetscInt n_field_dof_tot = 0;
  for (PetscInt f = 0; f < n_field; ++f) {
    PetscCall(PetscSectionSetFieldComponents(sec, f, n_field_comps[f]));
    for (PetscInt c = 0; c < n_field_comps[f]; ++c, ++n_field_dof_tot) {
      PetscCall(PetscSectionSetComponentName(sec, f, c, comp_names[c]));
    }
  }

  // set the number of degrees of freedom in each cell
  PetscInt c_start, c_end;  // starting and ending cell points
  PetscCall(DMPlexGetHeightStratum(rdy->dm, 0, &c_start, &c_end));
  PetscCall(PetscSectionSetChart(sec, c_start, c_end));
  for (PetscInt c = c_start; c < c_end; ++c) {
    for (PetscInt f = 0; f < n_field; ++f) {
      PetscCall(PetscSectionSetFieldDof(sec, c, f, n_field_comps[f]));
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

  // mark boundary edges so we can enforce reflecting BCs on them if needed
  {
    DMLabel boundary_edges;
    PetscCall(DMCreateLabel(rdy->dm, "boundary_edges"));
    PetscCall(DMGetLabel(rdy->dm, "boundary_edges", &boundary_edges));
    PetscCall(DMPlexMarkBoundaryFaces(rdy->dm, 1, boundary_edges));
  }

  // distribute the mesh across processes
  {
    DM      dm_dist;
    PetscSF sfMigration;
    PetscCall(DMPlexDistribute(rdy->dm, 1, &sfMigration, &dm_dist));
    if (dm_dist) {
      VecType vec_type;
      PetscCall(DMGetVecType(rdy->dm, &vec_type));
      PetscCall(DMSetVecType(dm_dist, vec_type));
      PetscCall(DMDestroy(&rdy->dm));
      rdy->dm = dm_dist;
      PetscCall(DMPlexSetMigrationSF(rdy->dm, sfMigration));
      PetscCall(PetscSFDestroy(&sfMigration));
    }
  }

  PetscCall(DMViewFromOptions(rdy->dm, NULL, "-dm_view"));

  PetscFunctionReturn(0);
}

// retrieves the index of a material using its name
static PetscErrorCode FindMaterial(RDy rdy, const char *name, PetscInt *index) {
  PetscFunctionBegin;

  // Currently, we do a linear search on the name of the material, which is O(N)
  // for N regions. If this is too slow, we can sort the conditions by name and
  // use binary search, which is O(log2 N).
  *index = -1;
  for (PetscInt i = 0; i < rdy->config.num_materials; ++i) {
    if (!strcmp(rdy->config.materials[i].name, name)) {
      *index = i;
      break;
    }
  }

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

// initializes mesh region data
static PetscErrorCode InitRegions(RDy rdy) {
  PetscFunctionBegin;

  // Count and fetch regions.
  PetscInt c_start, c_end;  // starting and ending cell points
  PetscCall(DMPlexGetHeightStratum(rdy->dm, 0, &c_start, &c_end));
  DMLabel label;
  PetscCall(DMGetLabel(rdy->dm, "Cell Sets", &label));
  if (label) {  // found regions (cell sets) in the grid
    PetscCall(DMLabelGetNumValues(label, &rdy->num_regions));
    PetscCheck(rdy->num_regions <= MAX_NUM_REGIONS, rdy->comm, PETSC_ERR_USER, "Number of regions in mesh (%d) exceeds MAX_NUM_REGIONS (%d)",
               rdy->num_regions, MAX_NUM_REGIONS);

    // fetch region IDs
    IS region_id_is;
    PetscCall(DMLabelGetValueIS(label, &region_id_is));
    const PetscInt *region_ids;
    PetscCall(ISGetIndices(region_id_is, &region_ids));

    // allocate and set region data
    PetscCall(RDyAlloc(PetscInt, rdy->num_regions, &rdy->region_ids));
    PetscCall(RDyAlloc(RDyRegion, rdy->num_regions, &rdy->regions));
    for (PetscInt r = 0; r < rdy->num_regions; ++r) {
      PetscInt region_id = region_ids[r];
      IS       cell_is;  // cell index space
      PetscCall(DMLabelGetStratumIS(label, region_id, &cell_is));
      if (cell_is) {
        RDyRegion *region  = &rdy->regions[r];
        rdy->region_ids[r] = region_id;

        PetscInt num_cells;
        PetscCall(ISGetLocalSize(cell_is, &num_cells));
        if (num_cells > 0) {
          RDyLogDebug(rdy, "  Found region %d (%d cells)", region_id, num_cells);
          region->num_cells = num_cells;
          PetscCall(RDyAlloc(PetscInt, region->num_cells, &region->cell_ids));
        }
        const PetscInt *cell_ids;
        PetscCall(ISGetIndices(cell_is, &cell_ids));
        for (PetscInt i = 0; i < num_cells; ++i) {
          region->cell_ids[i] = cell_ids[i] - c_start;
        }
        PetscCall(ISRestoreIndices(cell_is, &cell_ids));
        PetscCall(ISDestroy(&cell_is));
      }
    }
    PetscCall(ISRestoreIndices(region_id_is, &region_ids));
    PetscCall(ISDestroy(&region_id_is));
  } else {
    // If we didn't find any regions, we'd better have a file from which to
    // read initial conditions.
    PetscCheck(strlen(rdy->config.initial_conditions.domain.file), rdy->comm, PETSC_ERR_USER,
               "No regions (cell sets) found in grid, and no initial conditions file given! "
               "Cannot assign initial conditions for the given grid.");
  }

  PetscFunctionReturn(0);
}

// from Jed's commit (fd7c843):
//
// -dm_refine evidently completes boundary labels as a side-effect (so
// "Face Label" is applied also to vertices). I borrowed a utility from
// plexceed.c that we'll be able to delete later (by letting PETSc do all
// this setup).
// from plexceed.c
static PetscErrorCode DMGetPoints_Private(DM dm, DMLabel domainLabel, PetscInt labelVal, PetscInt height, IS *pointIS) {
  PetscInt depth;
  DMLabel  depthLabel;
  IS       depthIS;

  PetscFunctionBegin;
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMPlexGetDepthLabel(dm, &depthLabel));
  PetscCall(DMLabelGetStratumIS(depthLabel, depth - height, &depthIS));
  if (domainLabel) {
    IS domainIS;

    PetscCall(DMLabelGetStratumIS(domainLabel, labelVal, &domainIS));
    if (domainIS) {  // domainIS is non-empty
      PetscCall(ISIntersect(depthIS, domainIS, pointIS));
      PetscCall(ISDestroy(&domainIS));
    } else {  // domainIS is NULL (empty)
      *pointIS = NULL;
    }
    PetscCall(ISDestroy(&depthIS));
  } else {
    *pointIS = depthIS;
  }
  PetscFunctionReturn(0);
}

// initializes mesh boundary data
static PetscErrorCode InitBoundaries(RDy rdy) {
  PetscFunctionBegin;

  // Extract edges on the domain boundary.
  DMLabel boundary_edge_label;
  PetscCall(DMGetLabel(rdy->dm, "boundary_edges", &boundary_edge_label));
  IS boundary_edge_is;
  // FIXME: uncomment this line and delete the following one when PETSc's -dm_refine
  // FIXME: option has been updated
  // PetscCall(DMLabelGetStratumIS(boundary_edge_label, 1, &boundary_edge_is));
  PetscCall(DMGetPoints_Private(rdy->dm, boundary_edge_label, 1, 1, &boundary_edge_is));
  PetscBool boundary_edge_present = (boundary_edge_is != NULL);

  // Keep track of whether edges on the domain boundary have been assigned to
  // any boundaries.
  IS unassigned_edges_is;
  if (boundary_edge_present) {
    ISDuplicate(boundary_edge_is, &unassigned_edges_is);
  }
  PetscInt unassigned_edge_boundary_id = 0;  // boundary ID for unassigned edges

  // Count boundaries. We rely on face sets in our grids to express
  // boundary conditions. All edges on the domain boundary not assigned to other
  // boundaries are assigned to a special boundary to which we apply reflecting
  // boundary conditions.
  PetscInt e_start, e_end;  // starting and ending edge points
  DMPlexGetDepthStratum(rdy->dm, 1, &e_start, &e_end);
  DMLabel label;
  PetscCall(DMGetLabel(rdy->dm, "Face Sets", &label));
  PetscInt        num_boundaries_in_file = 0;
  IS              boundary_id_is;
  const PetscInt *boundary_ids;
  if (label) {  // found face sets!
    PetscCall(DMLabelGetNumValues(label, &num_boundaries_in_file));
    PetscCheck(num_boundaries_in_file <= MAX_NUM_BOUNDARIES, rdy->comm, PETSC_ERR_USER,
               "Number of boundaries in mesh (%d) exceeds MAX_NUM_BOUNDARIES (%d)", num_boundaries_in_file, MAX_NUM_BOUNDARIES);

    // fetch boundary IDs
    PetscCall(DMLabelGetValueIS(label, &boundary_id_is));
    PetscCall(ISGetIndices(boundary_id_is, &boundary_ids));

    for (PetscInt b = 0; b < num_boundaries_in_file; ++b) {
      PetscInt boundary_id = boundary_ids[b];
      IS       edge_is;
      // FIXME: uncomment this line and delete the following one when PETSc's -dm_refine
      // FIXME: option has been updated
      // PetscCall(DMLabelGetStratumIS(label, boundary_id, &edge_is));
      PetscCall(DMGetPoints_Private(rdy->dm, label, boundary_id, 1, &edge_is));
      if (edge_is) {
        PetscInt num_edges;
        PetscCall(ISGetLocalSize(edge_is, &num_edges));
        RDyLogDebug(rdy, "  Found boundary %d (%d edges)", boundary_id, num_edges);

        // we can't use this boundary ID for unassigned edges
        if (unassigned_edge_boundary_id == boundary_id) ++unassigned_edge_boundary_id;

        // intersect this IS with our domain boundary IS to produce the edges
        // to subtract from our unassigned edge IS
        IS assigned_edges_is, new_unassigned_edges_is;
        PetscCall(ISIntersect(edge_is, boundary_edge_is, &assigned_edges_is));
        PetscCall(ISDifference(unassigned_edges_is, assigned_edges_is, &new_unassigned_edges_is));
        ISDestroy(&assigned_edges_is);
        ISDestroy(&unassigned_edges_is);
        unassigned_edges_is = new_unassigned_edges_is;

        ++rdy->num_boundaries;
      }
      PetscCall(ISDestroy(&edge_is));
    }
    PetscCall(ISRestoreIndices(boundary_id_is, &boundary_ids));
    PetscCall(ISDestroy(&boundary_id_is));
  }

  // add an additional boundary for unassigned boundary edges if needed
  PetscInt num_unassigned_edges = 0;
  if (boundary_edge_present) {
    PetscCall(ISGetLocalSize(unassigned_edges_is, &num_unassigned_edges));
  }
  if (num_unassigned_edges > 0) {
    RDyLogDebug(rdy, "Adding boundary %d for %d unassigned boundary edges", unassigned_edge_boundary_id, num_unassigned_edges);
    if (!label) {
      // create a "Face Sets" label if one doesn't already exist
      PetscCall(DMCreateLabel(rdy->dm, "Face Sets"));
      PetscCall(DMGetLabel(rdy->dm, "Face Sets", &label));
    }
    // add these edges to a new boundary with the given ID
    PetscCall(DMLabelSetStratumIS(label, unassigned_edge_boundary_id, unassigned_edges_is));
    ++rdy->num_boundaries;
  }
  if (boundary_edge_present) {
    PetscCall(ISDestroy(&boundary_edge_is));
    PetscCall(ISDestroy(&unassigned_edges_is));
  }

  // allocate resources for boundaries
  PetscCall(RDyAlloc(PetscInt, rdy->num_boundaries, &rdy->boundary_ids));
  PetscCall(RDyAlloc(RDyBoundary, rdy->num_boundaries, &rdy->boundaries));

  // now fetch boundary edge IDs
  if (label) {
    // fetch boundary IDs once again
    PetscCall(DMLabelGetValueIS(label, &boundary_id_is));
    PetscCall(ISGetIndices(boundary_id_is, &boundary_ids));

    for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
      PetscInt boundary_id = (b < num_boundaries_in_file) ? boundary_ids[b] : unassigned_edge_boundary_id;
      IS       edge_is;  // edge index space//
      // FIXME: uncomment this line and delete the following one when PETSc's -dm_refine
      // FIXME: option has been updated
      // PetscCall(DMLabelGetStratumIS(label, boundary_id, &edge_is));
      PetscCall(DMGetPoints_Private(rdy->dm, label, boundary_id, 1, &edge_is));
      if (edge_is) {
        RDyBoundary *boundary = &rdy->boundaries[b];
        rdy->boundary_ids[b]  = boundary_id;

        // find the number of edges for this boundary
        PetscInt num_edges;
        PetscCall(ISGetLocalSize(edge_is, &num_edges));
        if (num_edges > 0) {
          boundary->num_edges = num_edges;
          PetscCall(RDyAlloc(PetscInt, boundary->num_edges, &boundary->edge_ids));
        }

        // extract edge IDs
        const PetscInt *edge_ids;
        PetscCall(ISGetIndices(edge_is, &edge_ids));
        for (PetscInt i = 0; i < num_edges; ++i) {
          boundary->edge_ids[i] = edge_ids[i] - e_start;
        }
        PetscCall(ISRestoreIndices(edge_is, &edge_ids));
        PetscCall(ISDestroy(&edge_is));
      }
    }
    PetscCall(ISRestoreIndices(boundary_id_is, &boundary_ids));
    PetscCall(ISDestroy(&boundary_id_is));
  }

  // make sure we have at least one region and boundary across all mpi ranks
  PetscInt num_global_boundaries = 0;
  MPI_Allreduce(&rdy->num_boundaries, &num_global_boundaries, 1, MPI_INT, MPI_SUM, rdy->comm);
  PetscCheck(num_global_boundaries > 0, rdy->comm, PETSC_ERR_USER, "No boundaries were found in the grid!");

  PetscFunctionReturn(0);
}

// reads data for a single DOF from a binary file into a Vec
static PetscErrorCode ReadOneDOFVecFromFile(RDy rdy, const char filename[], Vec *local) {
  PetscFunctionBegin;

  PetscViewer viewer;
  PetscCall(PetscViewerBinaryOpen(rdy->comm, filename, FILE_MODE_READ, &viewer));

  // create a naturally-ordered vector with a stride equal to the number of
  Vec natural, global;

  PetscCall(DMPlexCreateNaturalVector(rdy->aux_dm, &natural));
  PetscCall(DMCreateGlobalVector(rdy->aux_dm, &global));
  PetscCall(DMCreateLocalVector(rdy->aux_dm, local));

  // load the properties into the vector and copy them into place
  PetscCall(VecLoad(natural, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  // scatter natural-to-global
  PetscCall(DMPlexNaturalToGlobalBegin(rdy->aux_dm, natural, global));
  PetscCall(DMPlexNaturalToGlobalEnd(rdy->aux_dm, natural, global));

  // scatter global-to-local
  PetscCall(DMGlobalToLocalBegin(rdy->aux_dm, global, INSERT_VALUES, *local));
  PetscCall(DMGlobalToLocalEnd(rdy->aux_dm, global, INSERT_VALUES, *local));

  PetscCall(VecDestroy(&natural));
  PetscCall(VecDestroy(&global));

  PetscFunctionReturn(0);
}

#define READ_MATERIAL_PROPERTY_FROM_FILE(rdy, property)                                                   \
  if (strlen(rdy->config.surface_composition.domain.files.property)) {                                    \
    Vec local;                                                                                            \
    PetscCall(ReadOneDOFVecFromFile(rdy, rdy->config.surface_composition.domain.files.property, &local)); \
    PetscScalar *x_ptr;                                                                                   \
    PetscCall(VecGetArray(local, &x_ptr));                                                                \
    for (PetscInt icell = 0; icell < rdy->mesh.num_cells; icell++) {                                      \
      rdy->materials_by_cell[icell].property = x_ptr[icell];                                              \
    }                                                                                                     \
    PetscCall(VecRestoreArray(local, &x_ptr));                                                            \
    PetscCall(VecDestroy(&local));                                                                        \
  }

// sets up materials
static PetscErrorCode InitMaterials(RDy rdy) {
  PetscFunctionBegin;

  // allocate storage for materials for cells
  PetscCall(RDyAlloc(RDyMaterial, rdy->mesh.num_cells, &rdy->materials_by_cell));

  // read material properties for the entire domain from files if given
  READ_MATERIAL_PROPERTY_FROM_FILE(rdy, manning);

  // set up region-wise material and override cell-wise materials if needed
  if (rdy->config.surface_composition.num_regions > 0) {
    // allocate storage for regional materials
    PetscCall(RDyAlloc(RDyMaterial, rdy->num_regions, &rdy->materials));

    // assign materials to each region as needed
    for (PetscInt r = 0; r < rdy->num_regions; ++r) {
      RDyMaterial *material         = &rdy->materials[r];
      PetscInt     region_id        = rdy->region_ids[r];
      PetscInt     mat_region_index = -1;
      for (PetscInt imat = 0; imat < rdy->config.surface_composition.num_regions; ++imat) {
        if (rdy->config.surface_composition.by_region[imat].id == region_id) {
          mat_region_index = imat;
          break;
        }
      }
      if (mat_region_index != -1) {
        // retrieve the material object for this region
        RDyMaterialSpec *mat_spec = &rdy->config.surface_composition.by_region[mat_region_index];
        PetscInt         mat_index;
        PetscCall(FindMaterial(rdy, mat_spec->material, &mat_index));
        RDyMaterial *mat = &rdy->config.materials[mat_index];

        // set the region's material properties
        *material = *mat;

        // set the material properties for all cells in the region
        RDyRegion *region = &rdy->regions[r];
        for (PetscInt c = 0; c < region->num_cells; ++c) {
          PetscInt cell                = region->cell_ids[c];
          rdy->materials_by_cell[cell] = *mat;
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

// sets up initial conditions
static PetscErrorCode InitInitialConditions(RDy rdy) {
  PetscFunctionBegin;

  // allocate storage for by-region initial conditions
  PetscCall(RDyAlloc(RDyCondition, rdy->num_regions, &rdy->initial_conditions));

  // assign іnitial conditions to each region as indicated in our config
  for (PetscInt r = 0; r < rdy->num_regions; ++r) {
    RDyCondition *ic              = &rdy->initial_conditions[r];
    PetscInt      region_id       = rdy->region_ids[r];
    PetscInt      ic_region_index = -1;
    for (PetscInt ic = 0; ic < rdy->config.initial_conditions.num_regions; ++ic) {
      if (rdy->config.initial_conditions.by_region[ic].id == region_id) {
        ic_region_index = ic;
        break;
      }
    }
    PetscCheck(ic_region_index != -1 || strlen(rdy->config.initial_conditions.domain.file), rdy->comm, PETSC_ERR_USER,
               "Region %d has no initial conditions!", region_id);

    if (ic_region_index != -1) {
      RDyConditionSpec *ic_spec = &rdy->config.initial_conditions.by_region[ic_region_index];

      PetscCheck(strlen(ic_spec->flow), rdy->comm, PETSC_ERR_USER, "Region %d has no initial flow condition!", region_id);
      PetscInt flow_index;
      PetscCall(FindFlowCondition(rdy, ic_spec->flow, &flow_index));
      RDyFlowCondition *flow_cond = &rdy->config.flow_conditions[flow_index];
      PetscCheck(flow_cond->type == CONDITION_DIRICHLET, rdy->comm, PETSC_ERR_USER,
                 "initial flow condition %s for region %d is not of dirichlet type!", flow_cond->name, region_id);
      ic->flow = flow_cond;

      if (rdy->config.physics.sediment) {
        PetscCheck(strlen(ic_spec->sediment), rdy->comm, PETSC_ERR_USER, "Region %d has no initial sediment condition!", region_id);
        PetscInt sed_index;
        PetscCall(FindSedimentCondition(rdy, ic_spec->sediment, &sed_index));
        RDySedimentCondition *sed_cond = &rdy->config.sediment_conditions[sed_index];
        PetscCheck(sed_cond->type == CONDITION_DIRICHLET, rdy->comm, PETSC_ERR_USER,
                   "initial sediment condition %s for region %d is not of dirichlet type!", sed_cond->name, region_id);
        ic->sediment = sed_cond;
      }
      if (rdy->config.physics.salinity) {
        PetscCheck(strlen(ic_spec->salinity), rdy->comm, PETSC_ERR_USER, "Region %d has no initial salinity condition!", region_id);
        PetscInt sal_index;
        PetscCall(FindSalinityCondition(rdy, ic_spec->salinity, &sal_index));
        RDySalinityCondition *sal_cond = &rdy->config.salinity_conditions[sal_index];
        PetscCheck(sal_cond->type == CONDITION_DIRICHLET, rdy->comm, PETSC_ERR_USER,
                   "initial salinity condition %s for region %d is not of dirichlet type!", sal_cond->name, region_id);
        ic->salinity = sal_cond;
      }
    }
  }
  PetscFunctionReturn(0);
}

// sets up sources
static PetscErrorCode InitSources(RDy rdy) {
  PetscFunctionBegin;
  if (rdy->config.sources.num_regions > 0) {
    // Allocate storage for sources
    PetscCall(RDyAlloc(RDyCondition, rdy->num_regions, &rdy->sources));

    // Assign sources to each region as needed.
    for (PetscInt r = 0; r < rdy->num_regions; ++r) {
      RDyCondition *src              = &rdy->sources[r];
      PetscInt      region_id        = rdy->region_ids[r];
      PetscInt      src_region_index = -1;
      for (PetscInt isrc = 0; isrc < rdy->config.sources.num_regions; ++isrc) {
        if (rdy->config.sources.by_region[isrc].id == region_id) {
          src_region_index = isrc;
          break;
        }
      }
      if (src_region_index != -1) {
        RDyConditionSpec *src_spec = &rdy->config.sources.by_region[src_region_index];
        if (strlen(src_spec->flow)) {
          PetscInt flow_index;
          PetscCall(FindFlowCondition(rdy, src_spec->flow, &flow_index));
          RDyFlowCondition *flow_cond = &rdy->config.flow_conditions[flow_index];
          PetscCheck(flow_cond->type == CONDITION_DIRICHLET, rdy->comm, PETSC_ERR_USER, "flow source %s for region %d is not of dirichlet type!",
                     flow_cond->name, region_id);
          src->flow = flow_cond;
        }

        if (rdy->config.physics.sediment && strlen(src_spec->sediment)) {
          PetscInt sed_index;
          PetscCall(FindSedimentCondition(rdy, src_spec->sediment, &sed_index));
          RDySedimentCondition *sed_cond = &rdy->config.sediment_conditions[sed_index];
          PetscCheck(sed_cond->type == CONDITION_DIRICHLET, rdy->comm, PETSC_ERR_USER, "sediment source %s for region %d is not of dirichlet type!",
                     sed_cond->name, region_id);
          src->sediment = sed_cond;
        }
        if (rdy->config.physics.salinity && strlen(src_spec->salinity)) {
          PetscInt sal_index;
          PetscCall(FindSalinityCondition(rdy, src_spec->salinity, &sal_index));
          RDySalinityCondition *sal_cond = &rdy->config.salinity_conditions[sal_index];
          PetscCheck(sal_cond->type == CONDITION_DIRICHLET, rdy->comm, PETSC_ERR_USER,
                     "initial salinity condition %s for region %d is not of dirichlet type!", sal_cond->name, region_id);
          src->salinity = sal_cond;
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

// sets up boundary conditions
static PetscErrorCode InitBoundaryConditions(RDy rdy) {
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
  PetscCall(RDyAlloc(RDyCondition, rdy->num_boundaries, &rdy->boundary_conditions));

  // Assign a boundary condition to each boundary.
  for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
    RDyCondition *bc                = &rdy->boundary_conditions[b];
    PetscInt      boundary_id       = rdy->boundary_ids[b];
    PetscInt      bc_boundary_index = -1;
    for (PetscInt ib = 0; ib < rdy->config.num_boundary_conditions; ++ib) {
      if (rdy->config.boundary_conditions[ib].id == boundary_id) {
        bc_boundary_index = ib;
        break;
      }
    }
    if (bc_boundary_index != -1) {
      RDyConditionSpec *bc_spec = &rdy->config.boundary_conditions[bc_boundary_index];

      // If no flow condition was specified for a boundary, we set it to our
      // reflecting flow condition.
      if (!strlen(bc_spec->flow)) {
        bc->flow = reflecting_flow;
      } else {
        PetscInt flow_index;
        PetscCall(FindFlowCondition(rdy, bc_spec->flow, &flow_index));
        bc->flow = &rdy->config.flow_conditions[flow_index];
      }

      if (rdy->config.physics.sediment) {
        PetscCheck(strlen(bc_spec->sediment), rdy->comm, PETSC_ERR_USER, "Boundary %d has no sediment boundary condition!", boundary_id);
        PetscInt sed_index;
        PetscCall(FindSedimentCondition(rdy, bc_spec->sediment, &sed_index));
        bc->sediment = &rdy->config.sediment_conditions[sed_index];
      }
      if (rdy->config.physics.salinity) {
        PetscCheck(strlen(bc_spec->salinity), rdy->comm, PETSC_ERR_USER, "Boundary %d has no salinity boundary condition!", boundary_id);
        PetscInt sal_index;
        PetscCall(FindSalinityCondition(rdy, bc_spec->salinity, &sal_index));
        bc->salinity = &rdy->config.salinity_conditions[sal_index];
      }
    } else {
      // set a reflecting BC on this boundary
      bc->flow = reflecting_flow;
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

  PetscCall(DMCreateGlobalVector(rdy->aux_dm, &rdy->WaterSrc));

  PetscInt n_dof;
  PetscCall(VecGetSize(rdy->X, &n_dof));

  // set up a TS solver
  PetscCall(TSCreate(rdy->comm, &rdy->ts));
  PetscCall(TSSetProblemType(rdy->ts, TS_NONLINEAR));
  switch (rdy->config.numerics.temporal) {
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

  PetscCheck(rdy->config.physics.flow.mode == FLOW_SWE, rdy->comm, PETSC_ERR_USER, "Only the 'swe' flow mode is currently supported.");
  PetscCall(InitSWE(rdy));  // initialize SWE physics
  PetscCall(TSSetRHSFunction(rdy->ts, rdy->R, RHSFunctionSWE, rdy));

  PetscCall(TSSetMaxSteps(rdy->ts, rdy->config.time.max_step));
  PetscCall(TSSetExactFinalTime(rdy->ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetSolution(rdy->ts, rdy->X));
  PetscCall(TSSetTime(rdy->ts, 0.0));
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
  if (strlen(rdy->config.initial_conditions.domain.file)) {  // read from file
    PetscViewer viewer;
    PetscCall(PetscViewerBinaryOpen(rdy->comm, rdy->config.initial_conditions.domain.file, FILE_MODE_READ, &viewer));
    Vec natural;
    PetscCall(DMPlexCreateNaturalVector(rdy->dm, &natural));
    PetscCall(VecLoad(natural, viewer));
    PetscCall(DMPlexNaturalToGlobalBegin(rdy->dm, natural, rdy->X));
    PetscCall(DMPlexNaturalToGlobalEnd(rdy->dm, natural, rdy->X));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(VecDestroy(&natural));
  }

  // now initialize or override initial conditions by looping over regions
  // and writing values for corresponding cells
  PetscInt n_local;
  PetscCall(VecGetLocalSize(rdy->X, &n_local));
  PetscScalar *x_ptr;
  PetscCall(VecGetArray(rdy->X, &x_ptr));
  for (PetscInt r = 0; r < rdy->num_regions; ++r) {
    RDyRegion    *region = &rdy->regions[r];
    RDyCondition *ic     = &rdy->initial_conditions[r];
    if (ic->flow) {
      for (PetscInt c = 0; c < region->num_cells; ++c) {
        PetscInt cell_id = region->cell_ids[c];
        if (3 * cell_id < n_local) {  // skip ghost cells
          x_ptr[3 * cell_id]     = ic->flow->height;
          x_ptr[3 * cell_id + 1] = ic->flow->momentum[0];
          x_ptr[3 * cell_id + 2] = ic->flow->momentum[1];
        }
      }
    }
  }
  PetscCall(VecRestoreArray(rdy->X, &x_ptr));

  PetscFunctionReturn(0);
}

/// Performs any setup needed by RDy, reading from the specified configuration
/// file.
PetscErrorCode RDySetup(RDy rdy) {
  PetscFunctionBegin;

  // note: default config values are specified in the YAML input schema!
  PetscCall(ReadConfigFile(rdy));

  // open the primary log file
  if (strlen(rdy->config.logging.file)) {
    PetscCall(PetscFOpen(rdy->comm, rdy->config.logging.file, "w", &rdy->log));
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

  RDyLogDebug(rdy, "Initializing regions and boundaries...");
  PetscCall(InitRegions(rdy));
  PetscCall(InitBoundaries(rdy));

  RDyLogDebug(rdy, "Initializing initial/boundary conditions and sources...");
  PetscCall(InitInitialConditions(rdy));
  PetscCall(InitSources(rdy));
  PetscCall(InitBoundaryConditions(rdy));

  RDyLogDebug(rdy, "Creating solvers and vectors...");
  PetscCall(CreateSolvers(rdy));

  RDyLogDebug(rdy, "Creating FV mesh...");
  // note: this must be done after global vectors are created so a global
  // note: section exists for the DM
  PetscCall(RDyMeshCreateFromDM(rdy->dm, &rdy->mesh));

  RDyLogDebug(rdy, "Initializing materials...");
  PetscCall(InitMaterials(rdy));

  RDyLogDebug(rdy, "Initializing solution data...");
  PetscCall(InitSolution(rdy));

  PetscFunctionReturn(0);
}
