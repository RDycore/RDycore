#include <private/rdycoreimpl.h>
#include <private/rdymemoryimpl.h>

static PetscErrorCode InitBoundaryFluxes(RDy rdy) {
  PetscFunctionBegin;

  // allocate per-boundary flux offsets
  PetscCall(RDyAlloc(PetscInt, rdy->num_boundaries + 1, &(rdy->time_series.boundary_fluxes.offsets)));

  // make sure the number of degrees of freedom is the same as the number of
  // boundary fluxes
  PetscInt ndof;
  PetscCall(VecGetBlockSize(rdy->X_local, &ndof));
  PetscCheck(ndof == 3, rdy->comm, PETSC_ERR_USER, "ndof must be the same as # of boundary fluxes (3)");

  // count the number of global boundary edges (excluding those for
  // auto-generated boundary conditions and those not locally owned)
  // and compute local flux offsets
  PetscInt num_boundary_edges = 0;
  for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
    RDyCondition bc = rdy->boundary_conditions[b];
    if (!bc.auto_generated) {
      RDyBoundary *boundary = &rdy->boundaries[b];
      for (PetscInt e = 0; e < boundary->num_edges; ++e) {
        PetscInt edge_id = boundary->edge_ids[e];
        PetscInt cell_id = rdy->mesh.edges.cell_ids[2 * edge_id];
        if (rdy->mesh.cells.is_local[cell_id]) ++num_boundary_edges;
      }
    }
    rdy->time_series.boundary_fluxes.offsets[b + 1] = ndof * num_boundary_edges;
  }
  rdy->time_series.boundary_fluxes.num_local_edges = num_boundary_edges;

  // determine the global number of boundary edges
  MPI_Allreduce(&num_boundary_edges, &rdy->time_series.boundary_fluxes.num_global_edges, 1, MPI_INT, MPI_SUM, rdy->comm);

  // allocate (local) boundary flux storage
  PetscCall(RDyAlloc(PetscReal, 3 * num_boundary_edges, &(rdy->time_series.boundary_fluxes.fluxes)));

  PetscFunctionReturn(0);
}

// Initializes time series data storage.
PetscErrorCode InitTimeSeries(RDy rdy) {
  PetscFunctionBegin;
  rdy->time_series           = (RDyTimeSeriesData){0};
  rdy->time_series.last_step = -1;
  if (rdy->config.output.time_series.boundary_fluxes > 0) {
    InitBoundaryFluxes(rdy);
    PetscCall(TSMonitorSet(rdy->ts, WriteTimeSeries, rdy, NULL));
  }
  PetscFunctionReturn(0);
}

// Accumulates boundary fluxes on the given boundary from the given array of
// fluxes on boundary edges.
PetscErrorCode AccumulateBoundaryFluxes(RDy rdy, RDyBoundary *boundary, PetscInt num_edges, PetscReal fluxes[num_edges][3]) {
  PetscFunctionBegin;
  RDyTimeSeriesData *time_series = &rdy->time_series;
  if (time_series->boundary_fluxes.fluxes) {
    // figure out whether this boundary has an auto-generated BC
    PetscInt  b              = boundary - rdy->boundaries;
    PetscBool auto_generated = rdy->boundary_conditions[b].auto_generated;

    // if not, accumulate fluxes locally
    if (!auto_generated) {
      PetscInt n = rdy->time_series.boundary_fluxes.offsets[b];
      for (PetscInt e = 0; e < boundary->num_edges; ++e) {
        PetscInt edge_id   = boundary->edge_ids[e];
        PetscInt cell_id   = rdy->mesh.edges.cell_ids[2 * edge_id];
        PetscReal edge_len = rdy->mesh.edges.lengths[edge_id];
        if (rdy->mesh.cells.is_local[cell_id]) {
          time_series->boundary_fluxes.fluxes[n].water_mass += edge_len * fluxes[e][0] * rdy->dt;
          time_series->boundary_fluxes.fluxes[n].x_momentum += edge_len * fluxes[e][1] * rdy->dt;
          time_series->boundary_fluxes.fluxes[n].y_momentum += edge_len * fluxes[e][2] * rdy->dt;
          ++n;
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode WriteBoundaryFluxes(RDy rdy, PetscInt step, PetscReal time) {
  PetscFunctionBegin;

  PetscInt num_local_edges  = rdy->time_series.boundary_fluxes.num_local_edges;
  PetscInt num_global_edges = rdy->time_series.boundary_fluxes.num_global_edges;

  // flux metadata (global edge ID, boundary ID, BC type)
  // (add a padding byte to the end to prevent a 0-length VLA in case we don't
  // have any local boundary edges)
  PetscInt num_md = 3;
  PetscInt local_flux_md[num_md * num_local_edges + 1];

  // flux data itself (mass, x-momentum, y-momentum, edge x normal, edge y normal)
  // (add a padding byte to the end to prevent a 0-length VLA in case we don't
  // have any local boundary edges)
  PetscInt  num_data = 5;
  PetscReal local_flux_data[num_data * num_local_edges + 1];

  // gather local metadata/data
  PetscInt n = 0;
  for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
    RDyCondition bc = rdy->boundary_conditions[b];
    if (!bc.auto_generated) {  // exclude auto-generated boundary conditions
      RDyBoundary *boundary = &rdy->boundaries[b];
      for (PetscInt e = 0; e < boundary->num_edges; ++e) {
        PetscInt edge_id = boundary->edge_ids[e];
        PetscInt cell_id = rdy->mesh.edges.cell_ids[2 * edge_id];
        if (rdy->mesh.cells.is_local[cell_id]) {
          local_flux_md[num_md * n]     = rdy->mesh.edges.global_ids[edge_id];
          local_flux_md[num_md * n + 1] = boundary->id;
          local_flux_md[num_md * n + 2] = bc.flow->type;

          local_flux_data[num_data * n]     = rdy->time_series.boundary_fluxes.fluxes[n].water_mass;
          local_flux_data[num_data * n + 1] = rdy->time_series.boundary_fluxes.fluxes[n].x_momentum;
          local_flux_data[num_data * n + 2] = rdy->time_series.boundary_fluxes.fluxes[n].y_momentum;
          RDyVector edge_normal             = rdy->mesh.edges.normals[edge_id];
          local_flux_data[num_data * n + 3] = edge_normal.V[0];
          local_flux_data[num_data * n + 4] = edge_normal.V[1];
          ++n;
        }
      }
    }
  }

  // gather local metadata/data into global arrays on the root process and
  // write them out
  if (rdy->rank == 0) {
    // we need the number of local edges on each proc to determine the send
    // counts and displacements to pass to MPI_Gatherv below
    PetscInt n_local_edges[rdy->nproc];
    MPI_Gather(&num_local_edges, 1, MPI_INT, n_local_edges, 1, MPI_INT, 0, rdy->comm);
    PetscInt n_recv_counts[rdy->nproc], n_recv_displs[rdy->nproc + 1];
    n_recv_displs[0] = 0;

    // local -> global flux metadata
    for (PetscInt p = 0; p < rdy->nproc; ++p) {
      n_recv_counts[p]     = num_md * n_local_edges[p];
      n_recv_displs[p + 1] = n_recv_displs[p] + n_recv_counts[p];
    }
    PetscInt global_flux_md[num_md * num_global_edges];
    MPI_Gatherv(local_flux_md, num_md * num_local_edges, MPI_INT, global_flux_md, n_recv_counts, n_recv_displs, MPI_INT, 0, rdy->comm);

    // local -> global flux data
    for (PetscInt p = 0; p < rdy->nproc; ++p) {
      n_recv_counts[p]     = num_data * n_local_edges[p];
      n_recv_displs[p + 1] = n_recv_displs[p] + n_recv_counts[p];
    }
    PetscReal global_flux_data[num_data * num_global_edges];
    MPI_Gatherv(local_flux_data, num_data * num_local_edges, MPI_DOUBLE, global_flux_data, n_recv_counts, n_recv_displs, MPI_DOUBLE, 0, rdy->comm);

    // append the data to the boundary fluxes file (or, if this is our first
    // step, overwrite the existing file)
    char dir[PETSC_MAX_PATH_LEN];
    PetscCall(GetOutputDir(rdy, dir));
    char path[PETSC_MAX_PATH_LEN];
    snprintf(path, PETSC_MAX_PATH_LEN - 1, "%s/boundary_fluxes.dat", dir);
    FILE *fp = NULL;
    if (step == 0) {  // write a header on the first step
      PetscCall(PetscFOpen(rdy->comm, path, "w", &fp));
      PetscCall(PetscFPrintf(rdy->comm, fp,
                             "#time\tedge_id\tboundary_id\tbc_type\twater_mass_flux\tx_momentum_flux\ty_momentum_flux\tnormal_x\tnormal_y\t\n"));
    } else {
      PetscCall(PetscFOpen(rdy->comm, path, "a", &fp));
    }
    for (PetscInt e = 0; e < num_global_edges; ++e) {
      PetscInt  global_edge_id = global_flux_md[num_md * e];
      PetscInt  boundary_id    = global_flux_md[num_md * e + 1];
      PetscInt  bc_type        = global_flux_md[num_md * e + 2];
      PetscReal water_mass     = global_flux_data[num_data * e];
      PetscReal x_momentum     = global_flux_data[num_data * e + 1];
      PetscReal y_momentum     = global_flux_data[num_data * e + 2];
      PetscReal x_normal       = global_flux_data[num_data * e + 3];
      PetscReal y_normal       = global_flux_data[num_data * e + 4];
      PetscCall(PetscFPrintf(rdy->comm, fp, "%e\t%d\t%d\t%d\t%e\t%e\t%e\t%e\t%e\n", time, global_edge_id, boundary_id, bc_type, water_mass,
                             x_momentum, y_momentum, x_normal, y_normal));
    }
    PetscCall(PetscFClose(rdy->comm, fp));
  } else {
    // send the root proc information that allows it to compute global numbers
    // of edges, and to gather global flux metadata/data
    MPI_Gather(&num_local_edges, 1, MPI_INT, NULL, 1, MPI_INT, 0, rdy->comm);
    MPI_Gatherv(local_flux_md, num_md * num_local_edges, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, rdy->comm);
    MPI_Gatherv(local_flux_data, num_data * num_local_edges, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, rdy->comm);
  }

  // zero the boundary fluxes so they can begin reaccumulating
  // NOTE that there are 3 fluxes (and not 5)
  if (rdy->time_series.boundary_fluxes.fluxes) {
    memset(rdy->time_series.boundary_fluxes.fluxes, 0, 3 * num_global_edges * sizeof(PetscReal));
  }

  PetscFunctionReturn(0);
}

// This monitoring function writes out all requested time series data.
PetscErrorCode WriteTimeSeries(TS ts, PetscInt step, PetscReal time, Vec X, void *ctx) {
  PetscFunctionBegin;

  RDy rdy = ctx;
  if ((step % rdy->config.output.time_series.boundary_fluxes == 0) && (step > rdy->time_series.last_step)) {
    PetscCall(WriteBoundaryFluxes(rdy, step, time));
    rdy->time_series.last_step = step;
  }

  PetscFunctionReturn(0);
}

// Free resources allocated to time series.
PetscErrorCode DestroyTimeSeries(RDy rdy) {
  PetscFunctionBegin;
  if (rdy->time_series.boundary_fluxes.fluxes) {
    RDyFree(rdy->time_series.boundary_fluxes.offsets);
    RDyFree(rdy->time_series.boundary_fluxes.fluxes);
  }
  PetscFunctionReturn(0);
}
