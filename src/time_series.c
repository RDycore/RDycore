#include <private/rdycoreimpl.h>
#include <private/rdymemoryimpl.h>

static PetscErrorCode InitBoundaryFluxes(RDy rdy) {
  PetscFunctionBegin;

  // count up boundary fluxes and allocate storage.
  PetscInt num_boundary_fluxes = 0;
  for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
    RDyCondition bc = rdy->boundary_conditions[b];
    if (!bc.auto_generated) {  // exclude auto-generated boundary conditions
      RDyBoundary *boundary = &rdy->boundaries[b];
      num_boundary_fluxes += boundary->num_edges;
    }
  }
  PetscCall(RDyAlloc(PetscReal, 3 * num_boundary_fluxes, &rdy->time_series.boundary_fluxes));

  PetscFunctionReturn(0);
}

// Initializes time series data storage.
PetscErrorCode InitTimeSeries(RDy rdy) {
  PetscFunctionBegin;
  if (rdy->config.output.time_series.boundary_fluxes > 0) {
    InitBoundaryFluxes(rdy);
    PetscCall(TSMonitorSet(rdy->ts, WriteTimeSeries, rdy, NULL));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode WriteBoundaryFluxes(RDy rdy, PetscReal time) {
  PetscFunctionBegin;
  char dir[PETSC_MAX_PATH_LEN];
  PetscCall(GetOutputDir(rdy, dir));
  char path[PETSC_MAX_PATH_LEN];
  snprintf(path, PETSC_MAX_PATH_LEN - 1, "%s/boundary_fluxes.dat", dir);
  FILE *fp;
  PetscCall(PetscFOpen(rdy->comm, path, "a", &fp));  // append to end of file
  PetscInt n = 0;                                    // offset index for time series data
  for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
    RDyCondition bc = rdy->boundary_conditions[b];
    if (!bc.auto_generated) {  // exclude auto-generated boundary conditions
      RDyBoundary *boundary = &rdy->boundaries[b];
      for (PetscInt e = 0; e < boundary->num_edges; ++e, ++n) {
        PetscInt         edge_id        = boundary->edge_ids[e];
        PetscInt         global_edge_id = rdy->mesh.edges.global_ids[edge_id];
        RDyVector        edge_normal    = rdy->mesh.edges.normals[edge_id];
        PetscInt         boundary_id    = rdy->boundary_ids[b];
        RDyConditionType bc_type        = bc.flow->type;
        PetscReal        water_mass     = rdy->time_series.boundary_fluxes[n].water_mass;
        PetscReal        x_momentum     = rdy->time_series.boundary_fluxes[n].x_momentum;
        PetscReal        y_momentum     = rdy->time_series.boundary_fluxes[n].y_momentum;

        // tab-delimited data format is:
        // 1. simulation time (in desired units)
        // 2. (global) index of boundary edge
        // 3. integer ID indicating the boundary to which the boundary edge belongs
        // 4. integer ID indicating the type of boundary condition on the boundary edge
        // 5. water mass flux
        // 6. x-momentum flux
        // 7. y-momentum flux
        // 8. boundary edge normal x component
        // 9. boundary edge normal y component
        PetscCall(PetscFPrintf(rdy->comm, fp, "%e\t%d\t%d\t%d\t%e\t%e\t%e\t%e\t%e\n", time, global_edge_id, boundary_id, (PetscInt)bc_type,
                               water_mass, x_momentum, y_momentum, edge_normal.V[0], edge_normal.V[2]));
      }

      // zero the boundary fluxes so they can begin reaccumulating
      memset(rdy->time_series.boundary_fluxes, 0, 3 * n * sizeof(PetscReal));
    }
  }
  PetscCall(PetscFClose(rdy->comm, fp));
  PetscFunctionReturn(0);
}

// This monitoring function writes out all requested time series data.
PetscErrorCode WriteTimeSeries(TS ts, PetscInt step, PetscReal time, Vec X, void *ctx) {
  PetscFunctionBegin;

  RDy rdy = ctx;
  if (step % rdy->config.output.time_series.boundary_fluxes == 0) {
    PetscCall(WriteBoundaryFluxes(rdy, time));
  }

  PetscFunctionReturn(0);
}

// Free resources allocated to time series.
PetscErrorCode DestroyTimeSeries(RDy rdy) {
  PetscFunctionBegin;
  if (rdy->time_series.boundary_fluxes) {
    RDyFree(rdy->time_series.boundary_fluxes);
  }
  PetscFunctionReturn(0);
}
