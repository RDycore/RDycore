#include <muParserDLL.h>
#include <petscdmceed.h>
#include <petscdmplex.h>
#include <petscsys.h>
#include <private/rdycoreimpl.h>
#include <private/rdydmimpl.h>
#include <private/rdyoperatorimpl.h>
#include <rdycore.h>
#include <stdio.h>      // for getchar()
#include <sys/types.h>  // for getpid()
#include <unistd.h>     // for getpid() and gethostname()

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
  PetscFunctionReturn(time_unit_strings[time_unit - 1]);
}

/// Converts the given time (expressed in the given units) to seconds.
/// @param [in] time the time as expressed in the given units
/// @param [in] time_unit the units in which the time is expressed
PetscReal ConvertTimeToSeconds(PetscReal time, RDyTimeUnit time_unit) {
  PetscFunctionBegin;

  PetscReal time_in_sec;
  switch (time_unit) {
    case RDY_TIME_SECONDS:
      time_in_sec = time;
      break;
    case RDY_TIME_MINUTES:
      time_in_sec = time * secs_in_min;
      break;
    case RDY_TIME_HOURS:
      time_in_sec = time * mins_in_hr * secs_in_min;
      break;
    case RDY_TIME_DAYS:
      time_in_sec = time * hrs_in_day * mins_in_hr * secs_in_min;
      break;
    case RDY_TIME_MONTHS:
      time_in_sec = time * days_in_mon * hrs_in_day * mins_in_hr * secs_in_min;
      break;
    case RDY_TIME_YEARS:
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
    case RDY_TIME_SECONDS:
      time_in_units = time;
      break;
    case RDY_TIME_MINUTES:
      time_in_units = time / secs_in_min;
      break;
    case RDY_TIME_HOURS:
      time_in_units = time / mins_in_hr / secs_in_min;
      break;
    case RDY_TIME_DAYS:
      time_in_units = time / hrs_in_day / mins_in_hr / secs_in_min;
      break;
    case RDY_TIME_MONTHS:
      time_in_units = time / days_in_mon / hrs_in_day / mins_in_hr / secs_in_min;
      break;
    case RDY_TIME_YEARS:
      time_in_units = time / days_in_yr / hrs_in_day / mins_in_hr / secs_in_min;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Unsupported time unit");
      break;
  }

  PetscFunctionReturn(time_in_units);
}

// overrides parameters with command line arguments
PetscErrorCode OverrideParameters(RDy rdy) {
  PetscFunctionBegin;

  if (rdy->dt <= 0.0) {
    // ѕet a default timestep if needed
    rdy->dt = ConvertTimeToSeconds(rdy->config.time.final_time, rdy->config.time.unit) / rdy->config.time.max_step;
  } else {
    // convert dt to seconds in any case
    rdy->dt = ConvertTimeToSeconds(rdy->dt, rdy->config.time.unit);
  }

  char ceed_resource[PETSC_MAX_PATH_LEN] = {0};
  PetscOptionsBegin(rdy->comm, NULL, "RDycore options", "");
  {
    PetscCall(PetscOptionsReal("-dt", "time step size (seconds)", "", rdy->dt, &rdy->dt, NULL));
    PetscCall(PetscOptionsString("-ceed", "Ceed resource (/cpu/self, /gpu/cuda, /gpu/hip, ...)", "", ceed_resource, ceed_resource,
                                 sizeof ceed_resource, NULL));
    PetscCall(PetscOptionsString("-restart", "restart from the given checkpoint file", "", rdy->config.restart.file, rdy->config.restart.file,
                                 sizeof rdy->config.restart.file, NULL));
  }
  PetscOptionsEnd();

  // enable CEED as needed
  PetscCallCEED(SetCeedResource(ceed_resource));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// retrieves the index of a flow condition using its name
static PetscErrorCode FindFlowCondition(RDy rdy, const char *name, PetscInt *index) {
  PetscFunctionBegin;

  // NOTE: linear search on N condition names is O(N) complexity; binary search
  // NOTE: would be O(log2 N) if N is uncomfortable
  *index = -1;
  for (PetscInt i = 0; i < rdy->config.num_flow_conditions; ++i) {
    if (!strcmp(rdy->config.flow_conditions[i].name, name)) {
      *index = i;
      break;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// retrieves the index of a sediment condition using its name
static PetscErrorCode FindSedimentCondition(RDy rdy, const char *name, PetscInt *index) {
  PetscFunctionBegin;

  // NOTE: could be optimized as above
  *index = -1;
  for (PetscInt i = 0; i < rdy->config.num_sediment_conditions; ++i) {
    if (!strcmp(rdy->config.sediment_conditions[i].name, name)) {
      *index = i;
      break;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// retrieves the index of a salinity condition using its name
static PetscErrorCode FindSalinityCondition(RDy rdy, const char *name, PetscInt *index) {
  PetscFunctionBegin;

  // NOTE: could be optimized as above
  *index = -1;
  for (PetscInt i = 0; i < rdy->config.num_salinity_conditions; ++i) {
    if (!strcmp(rdy->config.salinity_conditions[i].name, name)) {
      *index = i;
      break;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// initializes mesh region data
//   can be run after refinement and FV mesh construction
PetscErrorCode InitRegions(RDy rdy) {
  PetscFunctionBegin;

  // count and fetch regions
  PetscInt c_start, c_end;  // starting and ending cell points
  PetscCall(DMPlexGetHeightStratum(rdy->dm, 0, &c_start, &c_end));
  DMLabel label;
  PetscCall(DMGetLabel(rdy->dm, "Cell Sets", &label));
  // if we didn't find any regions, we can't perform the simulation
  PetscCheck(label, rdy->comm, PETSC_ERR_USER, "No regions (cell sets) found in grid! Cannot assign initial conditions.");
  PetscCall(DMLabelGetNumValues(label, &rdy->num_regions));
  PetscCheck(rdy->num_regions <= MAX_NUM_REGIONS, rdy->comm, PETSC_ERR_USER,
             "Number of regions in mesh (%" PetscInt_FMT ") exceeds MAX_NUM_REGIONS (%d)", rdy->num_regions, MAX_NUM_REGIONS);

  // fetch region IDs
  IS region_id_is;
  PetscCall(DMLabelGetValueIS(label, &region_id_is));
  const PetscInt *region_ids;
  PetscCall(ISGetIndices(region_id_is, &region_ids));

  // allocate and set region data
  PetscCall(PetscCalloc1(rdy->num_regions, &rdy->regions));
  for (PetscInt r = 0; r < rdy->num_regions; ++r) {
    PetscInt region_id = region_ids[r];
    IS       cell_is;  // cell index space
    PetscCall(DMLabelGetStratumIS(label, region_id, &cell_is));
    if (cell_is) {
      RDyRegion *region = &rdy->regions[r];
      region->index     = r;
      region->id        = region_id;

      // fish the region's name out of our region config data
      for (PetscInt r1 = 0; r1 < rdy->config.num_regions; ++r1) {
        if (rdy->config.regions[r1].grid_region_id == region_id) {
          strcpy(region->name, rdy->config.regions[r1].name);
          break;
        }
      }

      // read local cell IDs
      PetscInt num_local_cells;
      PetscCall(ISGetLocalSize(cell_is, &num_local_cells));
      if (num_local_cells > 0) {
        RDyLogDebug(rdy, "  Found region %" PetscInt_FMT " (%" PetscInt_FMT " cells)", region_id, num_local_cells);
        region->num_local_cells = num_local_cells;
        PetscCall(PetscCalloc1(region->num_local_cells, &region->cell_local_ids));

        const PetscInt *cell_ids;
        PetscCall(ISGetIndices(cell_is, &cell_ids));
        for (PetscInt i = 0; i < num_local_cells; ++i) {
          region->cell_local_ids[i] = cell_ids[i] - c_start;
        }
        PetscCall(ISRestoreIndices(cell_is, &cell_ids));

        // construct global IDs for owned cells
        region->num_owned_cells = 0;
        for (PetscInt i = 0; i < num_local_cells; ++i) {
          if (rdy->mesh.cells.is_owned[region->cell_local_ids[i]]) {
            ++region->num_owned_cells;
          }
        }
        PetscCall(PetscCalloc1(region->num_owned_cells, &region->owned_cell_global_ids));
        PetscInt k = 0;
        for (PetscInt i = 0; i < num_local_cells; ++i) {
          if (rdy->mesh.cells.is_owned[region->cell_local_ids[i]]) {
            region->owned_cell_global_ids[k] = rdy->mesh.cells.local_to_owned[region->cell_local_ids[i]];
            ++k;
          }
        }
      }

      PetscCall(ISDestroy(&cell_is));
    }
  }
  PetscCall(ISRestoreIndices(region_id_is, &region_ids));
  PetscCall(ISDestroy(&region_id_is));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// initializes mesh boundary data
//   can be run after refinement
PetscErrorCode InitBoundaries(RDy rdy) {
  PetscFunctionBegin;

  // extract edges on the domain boundary
  DMLabel boundary_edge_label;
  PetscCall(DMGetLabel(rdy->dm, "boundary_edges", &boundary_edge_label));
  IS boundary_edge_is;
  PetscCall(DMLabelGetStratumIS(boundary_edge_label, 1, &boundary_edge_is));
  PetscBool boundary_edge_present = (boundary_edge_is != NULL);

  // track whether edges on the domain boundary have been assigned to boundaries
  PetscInt e_start, e_end;  // starting and ending edge points
  DMPlexGetHeightStratum(rdy->dm, 1, &e_start, &e_end);

  IS       unassigned_edges_is;
  PetscInt num_edges_invalid = 0;

  // first remove all edges that are not valid; invalid edges are
  // - edges between two non-local(=ghost) cells, and
  // - a boundary edge of a non-local(=ghost) cell
  if (boundary_edge_present) {
    ISDuplicate(boundary_edge_is, &unassigned_edges_is);

    RDyMesh  *mesh  = &rdy->mesh;
    RDyEdges *edges = &mesh->edges;
    RDyCells *cells = &mesh->cells;

    PetscInt *invalid_idx;
    PetscCall(PetscCalloc1(mesh->num_edges, &invalid_idx));

    for (PetscInt iedge = 0; iedge < mesh->num_edges; iedge++) {
      PetscInt  cell_id_1  = edges->cell_ids[2 * iedge];  // this cell id will always be > -1
      PetscInt  cell_id_2  = edges->cell_ids[2 * iedge + 1];
      PetscBool edge_valid = PETSC_TRUE;

      // check if the edge is valid
      if (cell_id_2 >= 0) {
        edge_valid = (cells->is_owned[cell_id_1] || cells->is_owned[cell_id_2]);
      } else {
        edge_valid = cells->is_owned[cell_id_1];
      }

      if (!edge_valid) {
        invalid_idx[num_edges_invalid++] = iedge + e_start;
      }
    }

    if (num_edges_invalid) {  // remove invalid edges, if any
      IS invalid_is, new_unassigned_edges_is;

      PetscCall(ISCreateGeneral(PETSC_COMM_SELF, num_edges_invalid, invalid_idx, PETSC_COPY_VALUES, &invalid_is));
      PetscCall(ISDifference(unassigned_edges_is, invalid_is, &new_unassigned_edges_is));
      ISDestroy(&invalid_is);
      ISDestroy(&unassigned_edges_is);
      unassigned_edges_is = new_unassigned_edges_is;
    }

    PetscCall(PetscFree(invalid_idx));
  }
  PetscInt unassigned_edge_boundary_id = 0;  // boundary ID for unassigned edges

  // count boundaries relying on face sets in our grids to express
  // boundary conditions; all edges on the domain boundary not assigned to other
  // boundaries are assigned to a special boundary to which we apply reflecting
  // boundary conditions
  DMLabel label;
  PetscCall(DMGetLabel(rdy->dm, "Face Sets", &label));
  PetscInt        num_boundaries_in_file = 0;
  IS              boundary_id_is;
  const PetscInt *boundary_ids;
  if (label) {  // found face sets!
    PetscCall(DMLabelGetNumValues(label, &num_boundaries_in_file));
    PetscCheck(num_boundaries_in_file <= MAX_NUM_BOUNDARIES, rdy->comm, PETSC_ERR_USER,
               "Number of boundaries in mesh (%" PetscInt_FMT ") exceeds MAX_NUM_BOUNDARIES (%d)", num_boundaries_in_file, MAX_NUM_BOUNDARIES);

    // fetch boundary IDs
    PetscCall(DMLabelGetValueIS(label, &boundary_id_is));
    PetscCall(ISGetIndices(boundary_id_is, &boundary_ids));

    // before we proceed, check that all boundaries in our configuration
    // exist in the grid file
    if (rdy->config.num_boundaries > 0) {
      PetscMPIInt boundary_in_file[MAX_NUM_BOUNDARIES];
      memset(boundary_in_file, 0, sizeof(PetscMPIInt) * rdy->config.num_boundaries);
      for (PetscInt bc = 0; bc < rdy->config.num_boundaries; ++bc) {
        for (PetscInt b = 0; b < num_boundaries_in_file; ++b) {
          if (rdy->config.boundaries[bc].grid_boundary_id == boundary_ids[b]) {
            boundary_in_file[bc] = 1;
            break;
          }
        }
      }
      MPI_Allreduce(MPI_IN_PLACE, boundary_in_file, rdy->config.num_boundaries, MPI_INT, MPI_MAX, rdy->comm);
      for (PetscInt b = 0; b < rdy->config.num_boundaries; ++b) {
        PetscCheck(boundary_in_file[b] > 0, rdy->comm, PETSC_ERR_USER, "The boundary '%s' was not found in the grid file '%s'.",
                   rdy->config.boundaries[b].name, rdy->config.grid.file);
      }
    }

    // process boundary data
    for (PetscInt b = 0; b < num_boundaries_in_file; ++b) {
      PetscInt boundary_id = boundary_ids[b];
      IS       edge_is;
      PetscCall(DMLabelGetStratumIS(label, boundary_id, &edge_is));
      if (edge_is) {
        PetscInt num_edges;
        PetscCall(ISGetLocalSize(edge_is, &num_edges));
        RDyLogDebug(rdy, "  Found boundary %" PetscInt_FMT " (%" PetscInt_FMT " edges)", boundary_id, num_edges);

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
  } else {
    // no Face Tags label, but we might still need the Allreduce
    if (rdy->config.num_boundaries > 0) {
      PetscMPIInt boundary_in_file[MAX_NUM_BOUNDARIES];
      memset(boundary_in_file, 0, sizeof(PetscMPIInt) * rdy->config.num_boundaries);
      MPI_Allreduce(boundary_in_file, boundary_in_file, rdy->config.num_boundaries, MPI_INT, MPI_SUM, rdy->comm);
      for (PetscInt b = 0; b < rdy->config.num_boundaries; ++b) {
        PetscCheck(boundary_in_file[b] > 0, rdy->comm, PETSC_ERR_USER, "The boundary '%s' was not found in the grid file '%s'.",
                   rdy->config.boundaries[b].name, rdy->config.grid.file);
      }
    }
  }

  // add an additional boundary for unassigned boundary edges if needed
  PetscInt num_unassigned_edges = 0;
  if (boundary_edge_present) {
    PetscCall(ISGetLocalSize(unassigned_edges_is, &num_unassigned_edges));
  }
  if (num_unassigned_edges > 0) {
    RDyLogDebug(rdy, "Adding boundary %" PetscInt_FMT " for %" PetscInt_FMT " unassigned boundary edges", unassigned_edge_boundary_id,
                num_unassigned_edges);
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
  PetscCall(PetscCalloc1(rdy->num_boundaries, &rdy->boundaries));

  // now fetch boundary edge IDs
  if (label) {
    // fetch boundary IDs once again
    PetscCall(DMLabelGetValueIS(label, &boundary_id_is));
    PetscCall(ISGetIndices(boundary_id_is, &boundary_ids));

    for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
      PetscInt boundary_id = (b < num_boundaries_in_file) ? boundary_ids[b] : unassigned_edge_boundary_id;
      IS       edge_is;  // edge index space//
      PetscCall(DMLabelGetStratumIS(label, boundary_id, &edge_is));
      if (edge_is) {
        RDyBoundary *boundary = &rdy->boundaries[b];
        boundary->index       = b;
        boundary->id          = boundary_id;

        // fish the boundary's name out of our boundary config data
        for (PetscInt b1 = 0; b1 < rdy->config.num_boundaries; ++b1) {
          if (rdy->config.boundaries[b1].grid_boundary_id == boundary_id) {
            strcpy(boundary->name, rdy->config.boundaries[b1].name);
            break;
          }
        }

        // find the number of edges for this boundary
        PetscInt num_edges;
        PetscCall(ISGetLocalSize(edge_is, &num_edges));
        if (num_edges > 0) {
          boundary->num_edges = num_edges;
          PetscCall(PetscCalloc1(boundary->num_edges, &boundary->edge_ids));
        }

        // extract edge IDs
        const PetscInt *edge_ids;
        PetscCall(ISGetIndices(edge_is, &edge_ids));
        for (PetscInt i = 0; i < num_edges; ++i) {
          PetscCheck((edge_ids[i] >= e_start) && (edge_ids[i] < e_end), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
                     "Mesh point %" PetscInt_FMT " is not an edge. Likely the option -dm_plex_transform_label_match_strata is missing", edge_ids[i]);
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
  PetscMPIInt num_global_boundaries = 0;
  MPI_Allreduce(&rdy->num_boundaries, &num_global_boundaries, 1, MPI_INT, MPI_MAX, rdy->comm);
  PetscCheck(num_global_boundaries > 0, rdy->comm, PETSC_ERR_USER, "No boundaries were found in the grid!");

  PetscFunctionReturn(PETSC_SUCCESS);
}

#define READ_MATERIAL_PROPERTY(property, mat_props_spec, values)                                       \
  {                                                                                                    \
    Vec mat_prop_vec = NULL;                                                                           \
    if (mat_props_spec.property.file[0]) {                                                             \
      RDyReadOneDOFLocalVecFromBinaryFile(rdy, mat_props_spec.property.file, &mat_prop_vec);           \
    }                                                                                                  \
    for (PetscInt r = 0; r < rdy->num_regions; ++r) {                                                  \
      RDyRegion region = rdy->regions[r];                                                              \
      for (PetscInt isurf_comp = 0; isurf_comp < rdy->config.num_material_assignments; ++isurf_comp) { \
        RDySurfaceCompositionSpec surface_comp = rdy->config.surface_composition[isurf_comp];          \
        if (!strcmp(surface_comp.region, region.name)) {                                               \
          if (mat_prop_vec) {                                                                          \
            PetscScalar *prop_ptr;                                                                     \
            PetscCall(VecGetArray(mat_prop_vec, &prop_ptr));                                           \
            for (PetscInt c = 0; c < region.num_local_cells; ++c) {                                    \
              PetscInt cell = region.cell_local_ids[c];                                                \
              values[cell]  = prop_ptr[c];                                                             \
            }                                                                                          \
            PetscCall(VecRestoreArray(mat_prop_vec, &prop_ptr));                                       \
          } else {                                                                                     \
            /* set this material property for all cells in each matching region */                     \
            for (PetscInt c = 0; c < region.num_local_cells; ++c) {                                    \
              PetscInt cell = region.cell_local_ids[c];                                                \
              values[cell]  = mupEval(mat_props_spec.property.value);                                  \
            }                                                                                          \
          }                                                                                            \
        }                                                                                              \
      }                                                                                                \
    }                                                                                                  \
    if (mat_prop_vec) {                                                                                \
      PetscCall(VecDestroy(&mat_prop_vec));                                                            \
    }                                                                                                  \
  }

// sets up materials
//   unsafe for refinement if file is given for surface composition
static PetscErrorCode InitMaterialProperties(RDy rdy) {
  PetscFunctionBegin;

  // check that each region has a material assigned to it
  for (PetscInt r = 0; r < rdy->num_regions; ++r) {
    RDyRegion region           = rdy->regions[r];
    PetscInt  region_mat_index = -1;
    for (PetscInt isurf_comp = 0; isurf_comp < rdy->config.num_material_assignments; ++isurf_comp) {
      RDySurfaceCompositionSpec surf_comp = rdy->config.surface_composition[isurf_comp];
      if (!strcmp(surf_comp.region, region.name)) {
        for (PetscInt imat = 0; imat < rdy->config.num_materials; ++imat) {
          if (!strcmp(rdy->config.materials[imat].name, surf_comp.material)) {
            region_mat_index = imat;
            break;
          }
        }
        if (region_mat_index != -1) break;
      }
    }
    PetscCheck(region_mat_index != -1, rdy->comm, PETSC_ERR_USER, "Region '%s' has no assigned material!", region.name);
  }

  // read material properties in from regional specifications and/or files
  PetscReal *material_property_values[OPERATOR_NUM_MATERIAL_PROPERTIES];
  for (PetscInt p = 0; p < OPERATOR_NUM_MATERIAL_PROPERTIES; ++p) {
    PetscCall(PetscCalloc1(rdy->mesh.num_cells, &material_property_values[p]));
  }
  for (PetscInt imat = 0; imat < rdy->config.num_materials; ++imat) {
    RDyMaterialPropertiesSpec mat_props_spec = rdy->config.materials[imat].properties;
    READ_MATERIAL_PROPERTY(manning, mat_props_spec, material_property_values[OPERATOR_MANNINGS]);
  }

  // set the properties on the operator
  OperatorData material_property;
  for (PetscInt property = 0; property < OPERATOR_NUM_MATERIAL_PROPERTIES; ++property) {
    PetscCall(GetOperatorDomainMaterialProperty(rdy->operator, property, &material_property));
    for (PetscInt i = 0; i < rdy->mesh.num_cells; ++i) {
      if (rdy->mesh.cells.is_owned[i]) {
        PetscInt owned_cell                     = rdy->mesh.cells.local_to_owned[i];
        material_property.values[0][owned_cell] = material_property_values[property][i];
      }
    }
    PetscCall(RestoreOperatorDomainMaterialProperty(rdy->operator, OPERATOR_MANNINGS, &material_property));
    PetscCall(PetscFree(material_property_values[property]));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// sets up initial conditions
//   can be run after refinement
static PetscErrorCode InitInitialConditions(RDy rdy) {
  PetscFunctionBegin;

  // allocate storage for by-region initial conditions
  PetscCall(PetscCalloc1(rdy->num_regions, &rdy->initial_conditions));

  // assign іnitial conditions to each region as indicated in our config
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

    if (rdy->config.physics.sediment.num_classes) {
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

// sets up sources
//   can be run after refinement
static PetscErrorCode InitSources(RDy rdy) {
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
          PetscCheck(flow_cond->type == CONDITION_RUNOFF, rdy->comm, PETSC_ERR_USER, "flow source '%s' for region '%s' is not of runoff type!",
                     flow_cond->name, region.name);
          src->flow = flow_cond;
        }

        if (rdy->config.physics.sediment.num_classes && strlen(src_spec.sediment)) {
          PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "Extend InitSources for sediments.");
        }
        if (rdy->config.physics.salinity && strlen(src_spec.salinity)) {
          PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "Extend InitSources for salinity.");
        }
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// sets up boundary conditions
//   can be run after refinement
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

      if (rdy->config.physics.sediment.num_classes) {
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

// initializes solution vector data
//   unsafe for refinement if file is given with initial conditions
static PetscErrorCode InitFlowSolution(RDy rdy) {
  PetscFunctionBegin;

  PetscCall(VecZeroEntries(rdy->flow_u_global));

  // check that each region has an initial condition
  for (PetscInt r = 0; r < rdy->num_regions; ++r) {
    RDyRegion    region = rdy->regions[r];
    RDyCondition ic     = rdy->initial_conditions[r];
    PetscCheck(ic.flow, rdy->comm, PETSC_ERR_USER, "No initial condition specified for region '%s'", region.name);
  }

  // now initialize or override initial conditions for each region
  PetscInt n_local, ndof;
  PetscCall(VecGetLocalSize(rdy->flow_u_global, &n_local));
  PetscCall(VecGetBlockSize(rdy->flow_u_global, &ndof));
  PetscScalar *u_ptr;
  PetscCall(VecGetArray(rdy->flow_u_global, &u_ptr));

  // initialize flow conditions
  for (PetscInt f = 0; f < rdy->config.num_flow_conditions; ++f) {
    RDyFlowCondition flow_ic = rdy->config.flow_conditions[f];
    Vec              local   = NULL;
    if (flow_ic.file[0]) {  // read flow data from file
      PetscViewer viewer;
      PetscCall(PetscViewerBinaryOpen(rdy->comm, flow_ic.file, FILE_MODE_READ, &viewer));

      Vec natural, global;
      PetscCall(DMPlexCreateNaturalVector(rdy->flow_dm, &natural));
      PetscCall(DMCreateGlobalVector(rdy->flow_dm, &global));
      PetscCall(DMCreateLocalVector(rdy->flow_dm, &local));

      PetscCall(VecLoad(natural, viewer));
      PetscCall(PetscViewerDestroy(&viewer));

      // check the block size of the initial condition vector agrees with the block size of rdy->u_global
      PetscInt nblocks_nat;
      PetscCall(VecGetBlockSize(natural, &nblocks_nat));
      PetscCheck((ndof == nblocks_nat), rdy->comm, PETSC_ERR_USER,
                 "The block size of the initial condition ('%" PetscInt_FMT
                 "') "
                 "does not match with the number of DOFs ('%" PetscInt_FMT "')",
                 nblocks_nat, ndof);

      // scatter natural-to-global
      PetscCall(DMPlexNaturalToGlobalBegin(rdy->flow_dm, natural, global));
      PetscCall(DMPlexNaturalToGlobalEnd(rdy->flow_dm, natural, global));

      // scatter global-to-local
      PetscCall(DMGlobalToLocalBegin(rdy->flow_dm, global, INSERT_VALUES, local));
      PetscCall(DMGlobalToLocalEnd(rdy->flow_dm, global, INSERT_VALUES, local));

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
          for (PetscInt c = 0; c < region.num_local_cells; ++c) {
            PetscInt cell_local_id = region.cell_local_ids[c];
            PetscInt owned_cell_id = rdy->mesh.cells.local_to_owned[cell_local_id];
            if (rdy->mesh.cells.is_owned[cell_local_id]) {  // skip ghost cells
              for (PetscInt idof = 0; idof < ndof; idof++) {
                u_ptr[ndof * owned_cell_id + idof] = local_ptr[ndof * cell_local_id + idof];
              }
            }
          }
          PetscCall(VecRestoreArray(local, &local_ptr));
          PetscCall(VecDestroy(&local));
        } else {
          // FIXME: this assumes the shallow water equations!
          for (PetscInt c = 0; c < region.num_owned_cells; ++c) {
            PetscInt owned_cell_id          = region.owned_cell_global_ids[c];
            u_ptr[ndof * owned_cell_id]     = mupEval(flow_ic.height);
            u_ptr[ndof * owned_cell_id + 1] = mupEval(flow_ic.x_momentum);
            u_ptr[ndof * owned_cell_id + 2] = mupEval(flow_ic.y_momentum);
          }
        }
      }
    }
  }

  // TODO: salinity and sediment initial conditions go here.

  PetscCall(VecRestoreArray(rdy->flow_u_global, &u_ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// initializes solution vector data
//   unsafe for refinement if file is given with initial conditions
static PetscErrorCode InitSedimentSolution(RDy rdy) {
  PetscFunctionBegin;

  PetscCall(VecZeroEntries(rdy->sd_u_global));

  // check that each region has an initial condition
  for (PetscInt r = 0; r < rdy->num_regions; ++r) {
    RDyRegion    region = rdy->regions[r];
    RDyCondition ic     = rdy->initial_conditions[r];
    PetscCheck(ic.flow, rdy->comm, PETSC_ERR_USER, "No initial condition specified for region '%s'", region.name);
  }

  // now initialize or override initial conditions for each region
  PetscInt n_local, ndof;
  PetscCall(VecGetLocalSize(rdy->sd_u_global, &n_local));
  PetscCall(VecGetBlockSize(rdy->sd_u_global, &ndof));
  PetscScalar *u_ptr;
  PetscCall(VecGetArray(rdy->sd_u_global, &u_ptr));

  // initialize sediment conditions
  for (PetscInt f = 0; f < rdy->config.num_sediment_conditions; ++f) {
    RDySedimentCondition sediment_ic = rdy->config.sediment_conditions[f];
    Vec                  local       = NULL;
    if (sediment_ic.file[0]) {  // read sediment data from file
      PetscViewer viewer;
      PetscCall(PetscViewerBinaryOpen(rdy->comm, sediment_ic.file, FILE_MODE_READ, &viewer));

      Vec natural, global;
      PetscCall(DMPlexCreateNaturalVector(rdy->sd_dm, &natural));
      PetscCall(DMCreateGlobalVector(rdy->sd_dm, &global));
      PetscCall(DMCreateLocalVector(rdy->sd_dm, &local));

      PetscCall(VecLoad(natural, viewer));
      PetscCall(PetscViewerDestroy(&viewer));

      // check the block size of the initial condition vector agrees with the block size of rdy->u_global
      PetscInt nblocks_nat;
      PetscCall(VecGetBlockSize(natural, &nblocks_nat));
      PetscCheck((ndof == nblocks_nat), rdy->comm, PETSC_ERR_USER,
                 "The block size of the initial condition ('%" PetscInt_FMT
                 "') "
                 "does not match with the number of DOFs ('%" PetscInt_FMT "')",
                 nblocks_nat, ndof);

      // scatter natural-to-global
      PetscCall(DMPlexNaturalToGlobalBegin(rdy->sd_dm, natural, global));
      PetscCall(DMPlexNaturalToGlobalEnd(rdy->sd_dm, natural, global));

      // scatter global-to-local
      PetscCall(DMGlobalToLocalBegin(rdy->sd_dm, global, INSERT_VALUES, local));
      PetscCall(DMGlobalToLocalEnd(rdy->sd_dm, global, INSERT_VALUES, local));

      // free up memory
      PetscCall(VecDestroy(&natural));
      PetscCall(VecDestroy(&global));
    }

    // set regional sediment as needed
    for (PetscInt r = 0; r < rdy->num_regions; ++r) {
      RDyRegion    region = rdy->regions[r];
      RDyCondition ic     = rdy->initial_conditions[r];
      if (!strcmp(ic.sediment->name, sediment_ic.name)) {
        if (local) {
          PetscScalar *local_ptr;
          PetscCall(VecGetArray(local, &local_ptr));
          for (PetscInt c = 0; c < region.num_local_cells; ++c) {
            PetscInt cell_local_id = region.cell_local_ids[c];
            PetscInt owned_cell_id = rdy->mesh.cells.local_to_owned[cell_local_id];
            if (rdy->mesh.cells.is_owned[cell_local_id]) {  // skip ghost cells
              for (PetscInt idof = 0; idof < ndof; idof++) {
                u_ptr[ndof * owned_cell_id + idof] = local_ptr[ndof * cell_local_id + idof];
              }
            }
          }
          PetscCall(VecRestoreArray(local, &local_ptr));
          PetscCall(VecDestroy(&local));
        } else {
          PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER, "mupEval is unsupported for Sediments.");
          // FIXME: this assumes the shallow water equations!
          /*
          for (PetscInt c = 0; c < region.num_owned_cells; ++c) {
            PetscInt owned_cell_id          = region.owned_cell_global_ids[c];
            u_ptr[ndof * owned_cell_id]     = mupEval(flow_ic.height);
            u_ptr[ndof * owned_cell_id + 1] = mupEval(flow_ic.x_momentum);
            u_ptr[ndof * owned_cell_id + 2] = mupEval(flow_ic.y_momentum);
          }
          */
        }
      }
    }
  }

  PetscCall(VecRestoreArray(rdy->sd_u_global, &u_ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitFlowAndSedimentSolution(RDy rdy) {
  PetscFunctionBegin;

  PetscInt flow_ndof, sediment_ndof, soln_ndof, diags_ndof;

  PetscCall(VecGetBlockSize(rdy->flow_u_global, &flow_ndof));
  PetscCall(VecGetBlockSize(rdy->sd_u_global, &sediment_ndof));
  PetscCall(VecGetBlockSize(rdy->u_global, &soln_ndof));
  PetscCall(VecGetBlockSize(rdy->diags_vec, &diags_ndof));

  PetscCheck(soln_ndof = flow_ndof + sediment_ndof, rdy->comm, PETSC_ERR_USER,
             "Blocksize of flow (=%d) and sediment (=%d) Vec do not sum to blocksize of solution (=%d) Vec", flow_ndof, sediment_ndof, soln_ndof);

  // first, copy flow Vec into solution Vec
  for (PetscInt i = 0; i < flow_ndof; i++) {
    PetscCall(VecStrideGather(rdy->flow_u_global, i, rdy->diags_vec, INSERT_VALUES));
    PetscCall(VecStrideScatter(rdy->diags_vec, i, rdy->u_global, INSERT_VALUES));
  }

  // next, copy sediment Vec into solution Vec
  for (PetscInt i = 0; i < sediment_ndof; i++) {
    PetscCall(VecStrideGather(rdy->sd_u_global, i, rdy->diags_vec, INSERT_VALUES));
    PetscCall(VecStrideScatter(rdy->diags_vec, flow_ndof + i, rdy->u_global, INSERT_VALUES));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitSolution(RDy rdy) {
  PetscFunctionBegin;
  PetscCall(InitFlowSolution(rdy));

  if (rdy->config.physics.sediment.num_classes) {
    PetscCall(InitSedimentSolution(rdy));
    PetscCall(InitFlowAndSedimentSolution(rdy));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// initializes the operator given the information in rdy
PetscErrorCode InitOperator(RDy rdy) {
  PetscFunctionBegin;

  PetscCall(CreateOperator(&rdy->config, rdy->dm, &rdy->mesh, rdy->num_regions, rdy->regions, rdy->num_boundaries, rdy->boundaries,
                           rdy->boundary_conditions, &rdy->operator));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// This is the right-hand-side function used by our timestepping solver.
/// @param [in]    ts  the solver
/// @param [in]    t   the simulation time [seconds]
/// @param [in]    U   the global solution vector at time t
/// @param [out]   F   the global right hand side vector to be evaluated at time t
/// @param [inout] ctx a pointer to the operator representing the system being solved
static PetscErrorCode OperatorRHSFunction(TS ts, PetscReal t, Vec U, Vec F, void *ctx) {
  PetscFunctionBegin;

  RDy       rdy = ctx;
  DM        dm  = rdy->dm;
  Operator *op  = rdy->operator;

  PetscScalar dt;
  PetscCall(TSGetTimeStep(ts, &dt));

  PetscCall(VecZeroEntries(F));

  // populate the local U vector
  PetscCall(DMGlobalToLocalBegin(dm, U, INSERT_VALUES, rdy->u_local));
  PetscCall(DMGlobalToLocalEnd(dm, U, INSERT_VALUES, rdy->u_local));

  PetscCall(ResetOperatorDiagnostics(op));

  // compute the right hand side
  PetscCall(ApplyOperator(op, dt, rdy->u_local, F));
  if (0) {
    PetscInt nstep;
    PetscCall(TSGetStepNumber(ts, &nstep));

    const char *backend = (CeedEnabled()) ? "ceed" : "petsc";
    char        file[PETSC_MAX_PATH_LEN];
    snprintf(file, PETSC_MAX_PATH_LEN, "F_%s_nstep%" PetscInt_FMT "_N%d.dat", backend, nstep, rdy->nproc);

    PetscViewer viewer;
    PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, file, &viewer));
    PetscCall(VecView(F, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  // if debug-level logging is enabled, find the latest global maximum Courant number
  // and log it.
  if (rdy->config.logging.level >= LOG_DEBUG) {
    PetscCall(UpdateOperatorDiagnostics(op));
    OperatorDiagnostics diagnostics;
    PetscCall(GetOperatorDiagnostics(op, &diagnostics));

    PetscReal time;
    PetscInt  stepnum;
    PetscCall(TSGetTime(ts, &time));
    PetscCall(TSGetStepNumber(ts, &stepnum));
    const char *units = TimeUnitAsString(rdy->config.time.unit);

    RDyLogDebug(rdy, "[%" PetscInt_FMT "] Time = %f [%s] Max courant number %g", stepnum, ConvertTimeFromSeconds(time, rdy->config.time.unit), units,
                diagnostics.courant_number.max_courant_num);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode InitSolver(RDy rdy) {
  PetscFunctionBegin;

  PetscInt n_dof;
  PetscCall(VecGetSize(rdy->u_global, &n_dof));

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
  PetscCall(TSSetApplicationContext(rdy->ts, rdy));

  PetscCheck(rdy->config.physics.flow.mode == FLOW_SWE, rdy->comm, PETSC_ERR_USER, "Only the 'swe' flow mode is currently supported.");
  PetscCall(TSSetRHSFunction(rdy->ts, rdy->rhs, OperatorRHSFunction, rdy));

  if (!rdy->config.time.adaptive.enable) {
    PetscCall(TSSetMaxSteps(rdy->ts, rdy->config.time.max_step));
  }
  PetscCall(TSSetExactFinalTime(rdy->ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetSolution(rdy->ts, rdy->u_global));
  PetscCall(TSSetTime(rdy->ts, 0.0));
  PetscCall(TSSetTimeStep(rdy->ts, rdy->dt));

  // apply any solver-related options supplied on the command line
  PetscCall(TSSetFromOptions(rdy->ts));
  PetscCall(TSGetTimeStep(rdy->ts, &rdy->dt));  // just in case!

  PetscFunctionReturn(PETSC_SUCCESS);
}

// initialize the data on the right hand side of the boundary edges
static PetscErrorCode InitDirichletBoundaryConditions(RDy rdy) {
  PetscFunctionBegin;

  for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
    RDyBoundary       boundary      = rdy->boundaries[b];
    RDyCondition      boundary_cond = rdy->boundary_conditions[b];
    RDyFlowCondition *flow_bc       = boundary_cond.flow;

    PetscReal *boundary_values;
    PetscCall(PetscCalloc1(3 * boundary.num_edges, &boundary_values));
    switch (boundary_cond.flow->type) {
      case CONDITION_DIRICHLET:

        // initialize the relevant boundary values
        for (PetscInt e = 0; e < boundary.num_edges; ++e) {
          boundary_values[3 * e]     = mupEval(flow_bc->height);
          boundary_values[3 * e + 1] = mupEval(flow_bc->x_momentum);
          boundary_values[3 * e + 2] = mupEval(flow_bc->y_momentum);
        }
        PetscCall(RDySetDirichletBoundaryValues(rdy, boundary.index, boundary.num_edges, 3, boundary_values));
        break;
      case CONDITION_REFLECTING:
        break;
      case CONDITION_CRITICAL_OUTFLOW:
        break;
      default:
        PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "Invalid boundary condition encountered for boundary %" PetscInt_FMT "\n",
                   boundary.id);
    }
    PetscCall(PetscFree(boundary_values));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Initializes a homogeneous water source term.
/// @param rdy A RDy struct
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode InitSourceConditions(RDy rdy) {
  PetscFunctionBegin;

  if (rdy->config.num_sources > 0) {
    for (PetscInt r = 0; r < rdy->num_regions; r++) {
      RDyCondition src = rdy->sources[r];

      RDyFlowCondition *flow_src = src.flow;
      if (flow_src) {
        PetscCall(RDySetHomogeneousRegionalWaterSource(rdy, r, mupEval(flow_src->value)));
      }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// the name of the log file set by RDySetLogFile below
static char overridden_logfile_[PETSC_MAX_PATH_LEN] = {0};

/// Sets the name of the log file for RDycore. If this function is called
/// before RDySetup, the name passed to it overrides any log filename set in
/// the YAML config file.
/// @param [inout] rdy      the RDycore simulator
/// @param [in]    log_file the name of the log file written by RDycore
PetscErrorCode RDySetLogFile(RDy rdy, const char *filename) {
  PetscFunctionBegin;
  strncpy(overridden_logfile_, filename, PETSC_MAX_PATH_LEN - 1);
  overridden_logfile_[PETSC_MAX_PATH_LEN - 1] = '\0';
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Checks for the -pause option, which pauses RDycore and prints out the PIDs
/// of the MPI processes so that a debugger may be attached.
/// @param [in] rdy the RDycore simulator
PetscErrorCode PauseIfRequested(RDy rdy) {
  PetscFunctionBegin;

  static PetscBool already_paused = PETSC_FALSE;
  if (!already_paused) {
    PetscBool pause = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-pause", &pause, NULL));
    if (pause) {
      pid_t pid          = getpid();  // local process ID
      char  hostname[65] = {0};       // local hostname (64 characters + null terminator)
      gethostname(hostname, 64);
      PetscFPrintf(rdy->comm, stderr, "Pausing... press Enter to resume.\n");
      if (rdy->nproc > 1) {
        pid_t *pids;
        char  *hostnames;
        PetscCall(PetscCalloc1(rdy->nproc, &pids));
        PetscCall(PetscCalloc1(rdy->nproc * (MAX_NAME_LEN + 1), &hostnames));
        MPI_Gather(&pid, 1, MPI_INT, pids, 1, MPI_INT, 0, rdy->comm);
        MPI_Gather(hostname, MAX_NAME_LEN + 1, MPI_CHAR, hostnames, MAX_NAME_LEN + 1, MPI_CHAR, 0, rdy->comm);
        PetscFPrintf(rdy->comm, stderr, "  PIDs (host):\n");
        for (PetscMPIInt p = 0; p < rdy->nproc; ++p) {
          PetscFPrintf(rdy->comm, stderr, "    rank %d (%s): %d:\n", p, &hostnames[p * (MAX_NAME_LEN + 1)], pids[p]);
        }
        PetscCall(PetscFree(pids));
        PetscCall(PetscFree(hostnames));

        // wait for input on rank 0
        if (rdy->rank == 0) {
          getchar();
        }
        MPI_Barrier(rdy->comm);
      } else {
        PetscFPrintf(rdy->comm, stderr, "  PID on host %s: %d\n", hostname, pid);
        getchar();
      }
      already_paused = PETSC_TRUE;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Performs any setup needed by RDy, reading from the specified configuration
/// file and overwriting options with command-line arguments.
/// @param [inout] rdy the RDycore simulator
PetscErrorCode RDySetup(RDy rdy) {
  PetscFunctionBegin;

  PetscCall(PauseIfRequested(rdy));

  // note: default config values are specified in the YAML input schema!
  PetscCall(ReadConfigFile(rdy));

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

  // print configuration info
  PetscCall(PrintConfig(rdy));

  RDyLogDebug(rdy, "Creating DMs...");

  // create the primary DM that stores the mesh and solution vector
  rdy->soln_fields = (SectionFieldSpec){
      .num_fields            = 1,
      .num_field_components  = {3},
      .field_names           = {"Solution"},
      .field_component_names = {{
          "Height",
          "MomentumX",
          "MomentumY",
      }},
  };
  PetscCall(CreateDM(rdy));

  // create the auxiliary DM, which handles diagnostics and I/O
  rdy->diag_fields = (SectionFieldSpec){
      .num_fields           = 1,
      .num_field_components = {1},
      .field_names          = {"Parameter"},
  };
  PetscCall(CreateAuxiliaryDM(rdy));

  PetscCall(CreateFlowDM(rdy));

  if (rdy->config.physics.sediment.num_classes) {
    PetscCall(CreateSedimentDM(rdy));
    PetscCall(CreateCombinedDM(rdy));
  } else {
    rdy->flow_dm = rdy->dm;
  }

  // create global and local vectors
  PetscCall(CreateVectors(rdy));

  RDyLogDebug(rdy, "Creating FV mesh...");
  PetscCall(RDyMeshCreateFromDM(rdy->dm, &rdy->mesh));

  RDyLogDebug(rdy, "Initializing regions...");
  PetscCall(InitRegions(rdy));

  RDyLogDebug(rdy, "Initializing initial conditions and sources...");
  PetscCall(InitInitialConditions(rdy));
  PetscCall(InitSources(rdy));

  RDyLogDebug(rdy, "Initializing boundaries and boundary conditions...");
  PetscCall(InitBoundaries(rdy));
  PetscCall(InitBoundaryConditions(rdy));

  RDyLogDebug(rdy, "Initializing solution data...");
  PetscCall(InitSolution(rdy));

  RDyLogDebug(rdy, "Initializing operator...");
  PetscCall(InitOperator(rdy));

  RDyLogDebug(rdy, "Initializing material properties...");
  PetscCall(InitMaterialProperties(rdy));

  RDyLogDebug(rdy, "Initializing solver...");
  PetscCall(InitSolver(rdy));

  // make sure any Dirichlet boundary conditions are properly specified
  PetscCall(InitDirichletBoundaryConditions(rdy));

  // initialize the source terms
  PetscCall(InitSourceConditions(rdy));

  RDyLogDebug(rdy, "Initializing checkpoints...");
  PetscCall(InitCheckpoints(rdy));

  // if a restart has been requested, read the specified checkpoint file
  // and overwrite the necessary data
  if (rdy->config.restart.file[0]) {
    PetscCall(ReadCheckpointFile(rdy, rdy->config.restart.file));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
