#include <petscdmceed.h>
#include <private/rdycoreimpl.h>
#include <private/rdysweimpl.h>  // for CEED boundary flux accumulation

static PetscErrorCode GatherBoundaryFluxMetadata(RDy rdy) {
  PetscFunctionBegin;

  // allocate storage for local metadata
  PetscInt  num_md          = 3;
  PetscInt  num_local_edges = rdy->time_series.boundary_fluxes.num_local_edges[rdy->rank];
  PetscInt *local_flux_md;
  PetscCall(PetscCalloc1(num_md * num_local_edges + 1, &local_flux_md));

  // gather local metadata
  PetscInt n = 0;
  for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
    RDyCondition bc = rdy->boundary_conditions[b];
    if (!bc.auto_generated) {  // exclude auto-generated boundary conditions
      RDyBoundary boundary = rdy->boundaries[b];
      for (PetscInt e = 0; e < boundary.num_edges; ++e) {
        PetscInt edge_id = boundary.edge_ids[e];
        PetscInt cell_id = rdy->mesh.edges.cell_ids[2 * edge_id];
        if (rdy->mesh.cells.is_owned[cell_id]) {
          local_flux_md[num_md * n]     = rdy->mesh.edges.centroids[edge_id].X[0];
          local_flux_md[num_md * n + 1] = rdy->mesh.edges.centroids[edge_id].X[1];
          local_flux_md[num_md * n + 2] = bc.flow->type;
          ++n;
        }
      }
    }
  }

  // gather it on the root process (rank 0)
  if (rdy->rank == 0) {
    PetscMPIInt *global_flux_md, *n_recv_counts, *n_recv_displs;
    PetscInt     num_global_edges = rdy->time_series.boundary_fluxes.num_global_edges;
    PetscCall(PetscCalloc1(num_md * num_global_edges, &global_flux_md));
    PetscCall(PetscCalloc1(rdy->nproc, &n_recv_counts));
    PetscCall(PetscCalloc1(rdy->nproc + 1, &n_recv_displs));

    // local -> global flux metadata
    n_recv_displs[0] = 0;
    for (PetscInt p = 0; p < rdy->nproc; ++p) {
      n_recv_counts[p]     = num_md * rdy->time_series.boundary_fluxes.num_local_edges[p];
      n_recv_displs[p + 1] = n_recv_displs[p] + n_recv_counts[p];
    }
    MPI_Gatherv(local_flux_md, num_md * num_local_edges, MPI_INT, global_flux_md, n_recv_counts, n_recv_displs, MPI_INT, 0, rdy->comm);
    PetscCall(PetscCalloc1(num_md * num_global_edges, &rdy->time_series.boundary_fluxes.global_flux_md));
    for (PetscInt i = 0; i < num_md * num_global_edges; ++i) {
      rdy->time_series.boundary_fluxes.global_flux_md[i] = (PetscInt)global_flux_md[i];
    }
    PetscCall(PetscFree(global_flux_md));
    PetscCall(PetscFree(n_recv_counts));
    PetscCall(PetscFree(n_recv_displs));
  } else {
    // send the root proc the local flux metadata
    MPI_Gatherv(local_flux_md, num_md * num_local_edges, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, rdy->comm);
  }

  PetscCall(PetscFree(local_flux_md));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitBoundaryFluxes(RDy rdy) {
  PetscFunctionBegin;

  // allocate per-boundary flux offsets
  PetscCall(PetscCalloc1(rdy->num_boundaries + 1, &(rdy->time_series.boundary_fluxes.offsets)));

  // make sure the number of degrees of freedom is the same as the number of
  // boundary fluxes
  PetscInt ndof;
  PetscCall(VecGetBlockSize(rdy->u_local, &ndof));
  PetscCheck(ndof == 3, rdy->comm, PETSC_ERR_USER, "ndof must be the same as # of boundary fluxes (3)");

  // count the number of global boundary edges (excluding those for
  // auto-generated boundary conditions and those not locally owned)
  // and compute local flux offsets
  PetscInt num_boundary_edges = 0;
  for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
    RDyCondition bc = rdy->boundary_conditions[b];
    if (!bc.auto_generated) {
      RDyBoundary boundary = rdy->boundaries[b];
      for (PetscInt e = 0; e < boundary.num_edges; ++e) {
        PetscInt edge_id = boundary.edge_ids[e];
        PetscInt cell_id = rdy->mesh.edges.cell_ids[2 * edge_id];
        if (rdy->mesh.cells.is_owned[cell_id]) ++num_boundary_edges;
      }
    }
    rdy->time_series.boundary_fluxes.offsets[b + 1] = num_boundary_edges;
  }

  // gather per-process numbers of local boundary edges
  PetscMPIInt *num_local_edges;
  PetscCall(PetscCalloc1(rdy->nproc, &num_local_edges));
  MPI_Allgather(&num_boundary_edges, 1, MPI_INT, num_local_edges, 1, MPI_INT, rdy->comm);
  PetscCall(PetscCalloc1(rdy->nproc, &rdy->time_series.boundary_fluxes.num_local_edges));
  for (PetscInt p = 0; p < rdy->nproc; ++p) {
    rdy->time_series.boundary_fluxes.num_local_edges[p] = (PetscInt)num_local_edges[p];
  }
  PetscCall(PetscFree(num_local_edges));

  // determine the global number of boundary edges
  MPI_Allreduce(&num_boundary_edges, &rdy->time_series.boundary_fluxes.num_global_edges, 1, MPI_INT, MPI_SUM, rdy->comm);

  // determine global flux metadata on rank 0
  PetscCall(GatherBoundaryFluxMetadata(rdy));

  // allocate (local) boundary flux storage
  PetscCall(PetscCalloc1(num_boundary_edges, &(rdy->time_series.boundary_fluxes.fluxes)));

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

// accumulates boundary fluxes on the given boundary from the given array of
// fluxes on boundary edges
static PetscErrorCode AccumulateBoundaryFluxes(RDy rdy, RDyBoundary boundary, OperatorData boundary_fluxes) {
  PetscFunctionBegin;
  RDyTimeSeriesData *time_series = &rdy->time_series;
  if (time_series->boundary_fluxes.fluxes) {
    // if the boundary condition for this boundary is auto-generated,
    // accumulate fluxes locally
    if (!rdy->boundary_conditions[boundary.index].auto_generated) {
      PetscInt n = rdy->time_series.boundary_fluxes.offsets[boundary.index];
      for (PetscInt e = 0; e < boundary.num_edges; ++e) {
        PetscInt  edge_id  = boundary.edge_ids[e];
        PetscInt  cell_id  = rdy->mesh.edges.cell_ids[2 * edge_id];
        PetscReal edge_len = rdy->mesh.edges.lengths[edge_id];
        if (rdy->mesh.cells.is_owned[cell_id]) {
          // FIXME: this is specific to the shallow water equations
          time_series->boundary_fluxes.fluxes[n].water_mass += edge_len * boundary_fluxes.values[0][e];
          time_series->boundary_fluxes.fluxes[n].x_momentum += edge_len * boundary_fluxes.values[1][e];
          time_series->boundary_fluxes.fluxes[n].y_momentum += edge_len * boundary_fluxes.values[2][e];
          ++n;
        }
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode WriteBoundaryFluxes(RDy rdy, PetscInt step, PetscReal time) {
  PetscFunctionBegin;

  PetscInt num_local_edges  = rdy->time_series.boundary_fluxes.num_local_edges[rdy->rank];
  PetscInt num_global_edges = rdy->time_series.boundary_fluxes.num_global_edges;

  // flux data itself (mass, x-momentum, y-momentum, edge x normal, edge y normal)
  // (add a padding byte to the end to prevent a 0-length VLA in case we don't
  // have any local boundary edges)
  PetscInt   num_data = 5;
  PetscReal *local_flux_data;
  PetscCall(PetscCalloc1(num_data * num_local_edges + 1, &local_flux_data));

  // gather local data
  PetscInt n = 0;
  for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
    RDyCondition bc = rdy->boundary_conditions[b];
    if (!bc.auto_generated) {  // exclude auto-generated boundary conditions
      RDyBoundary boundary = rdy->boundaries[b];
      for (PetscInt e = 0; e < boundary.num_edges; ++e) {
        PetscInt edge_id = boundary.edge_ids[e];
        PetscInt cell_id = rdy->mesh.edges.cell_ids[2 * edge_id];
        if (rdy->mesh.cells.is_owned[cell_id]) {
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

  // gather local data into global arrays on the root process and
  // write them out
  if (rdy->rank == 0) {
    int          num_md         = 3;
    PetscInt    *global_flux_md = rdy->time_series.boundary_fluxes.global_flux_md;
    PetscMPIInt *n_recv_counts, *n_recv_displs;
    PetscCall(PetscCalloc1(rdy->nproc, &n_recv_counts));
    PetscCall(PetscCalloc1(rdy->nproc + 1, &n_recv_displs));

    n_recv_displs[0] = 0;
    for (PetscInt p = 0; p < rdy->nproc; ++p) {
      n_recv_counts[p]     = num_data * rdy->time_series.boundary_fluxes.num_local_edges[p];
      n_recv_displs[p + 1] = n_recv_displs[p] + n_recv_counts[p];
    }
    PetscReal *global_flux_data;
    PetscCall(PetscCalloc1(num_data * num_global_edges, &global_flux_data));
    MPI_Gatherv(local_flux_data, num_data * num_local_edges, MPI_DOUBLE, global_flux_data, n_recv_counts, n_recv_displs, MPI_DOUBLE, 0, rdy->comm);
    PetscCall(PetscFree(n_recv_counts));
    PetscCall(PetscFree(n_recv_displs));

    // append the data to the boundary fluxes file (or, if this is our first
    // step, overwrite the existing file)
    char output_dir[PETSC_MAX_PATH_LEN], prefix[PETSC_MAX_PATH_LEN], path[PETSC_MAX_PATH_LEN];
    PetscCall(GetOutputDirectory(rdy, output_dir));
    PetscCall(DetermineConfigPrefix(rdy, prefix));
    snprintf(path, PETSC_MAX_PATH_LEN - 1, "%s/%s-boundary_fluxes.dat", output_dir, prefix);
    FILE *fp = NULL;
    if (step == 0) {  // write a header on the first step
      PetscCall(PetscFOpen(rdy->comm, path, "w", &fp));
      PetscCall(
          PetscFPrintf(rdy->comm, fp, "#time\tedge_xc\tedge_yc\tbc_type\twater_mass_flux\tx_momentum_flux\ty_momentum_flux\tnormal_x\tnormal_y\t\n"));
    } else {
      PetscCall(PetscFOpen(rdy->comm, path, "a", &fp));
    }
    for (PetscInt e = 0; e < num_global_edges; ++e) {
      PetscReal edge_xc    = global_flux_md[num_md * e];
      PetscReal edge_yc    = global_flux_md[num_md * e + 1];
      PetscInt  bc_type    = global_flux_md[num_md * e + 2];
      PetscReal water_mass = global_flux_data[num_data * e];
      PetscReal x_momentum = global_flux_data[num_data * e + 1];
      PetscReal y_momentum = global_flux_data[num_data * e + 2];
      PetscReal x_normal   = global_flux_data[num_data * e + 3];
      PetscReal y_normal   = global_flux_data[num_data * e + 4];
      PetscCall(PetscFPrintf(rdy->comm, fp, "%e\t%f\t%f\t%" PetscInt_FMT "\t%e\t%e\t%e\t%e\t%e\n", time, edge_xc, edge_yc, bc_type, water_mass,
                             x_momentum, y_momentum, x_normal, y_normal));
    }
    PetscCall(PetscFree(global_flux_data));
    PetscCall(PetscFClose(rdy->comm, fp));
  } else {
    // send the root proc the local flux data
    MPI_Gatherv(local_flux_data, num_data * num_local_edges, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, rdy->comm);
  }
  PetscCall(PetscFree(local_flux_data));

  // zero the boundary fluxes so they can begin reaccumulating
  // NOTE that there are 3 fluxes (and not 5)
  if (rdy->time_series.boundary_fluxes.fluxes) {
    for (PetscInt e = 0; e < rdy->time_series.boundary_fluxes.offsets[rdy->num_boundaries]; e++) {
      rdy->time_series.boundary_fluxes.fluxes[e].water_mass = 0.0;
      rdy->time_series.boundary_fluxes.fluxes[e].x_momentum = 0.0;
      rdy->time_series.boundary_fluxes.fluxes[e].y_momentum = 0.0;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// This monitoring function writes out all requested time series data.
PetscErrorCode WriteTimeSeries(TS ts, PetscInt step, PetscReal time, Vec X, void *ctx) {
  PetscFunctionBegin;

  RDy rdy = ctx;
  if ((step % rdy->config.output.time_series.boundary_fluxes == 0) && (step > rdy->time_series.last_step)) {
    for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
      RDyBoundary boundary = rdy->boundaries[b];

      OperatorData boundary_fluxes;
      PetscCall(GetOperatorBoundaryFluxes(rdy->operator, boundary, &boundary_fluxes));
      PetscCall(AccumulateBoundaryFluxes(rdy, boundary, boundary_fluxes));
      PetscCall(RestoreOperatorBoundaryFluxes(rdy->operator, boundary, &boundary_fluxes));
    }
    PetscCall(WriteBoundaryFluxes(rdy, step, time));
    rdy->time_series.last_step = step;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Free resources allocated to time series.
PetscErrorCode DestroyTimeSeries(RDy rdy) {
  PetscFunctionBegin;
  if (rdy->time_series.boundary_fluxes.fluxes) {
    PetscFree(rdy->time_series.boundary_fluxes.num_local_edges);
    PetscFree(rdy->time_series.boundary_fluxes.global_flux_md);
    PetscFree(rdy->time_series.boundary_fluxes.offsets);
    PetscFree(rdy->time_series.boundary_fluxes.fluxes);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
