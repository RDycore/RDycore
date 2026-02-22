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
    PetscCallMPI(MPI_Gatherv(local_flux_md, num_md * num_local_edges, MPI_INT, global_flux_md, n_recv_counts, n_recv_displs, MPI_INT, 0, rdy->comm));
    PetscCall(PetscCalloc1(num_md * num_global_edges, &rdy->time_series.boundary_fluxes.global_flux_md));
    for (PetscInt i = 0; i < num_md * num_global_edges; ++i) {
      rdy->time_series.boundary_fluxes.global_flux_md[i] = (PetscInt)global_flux_md[i];
    }
    PetscCall(PetscFree(global_flux_md));
    PetscCall(PetscFree(n_recv_counts));
    PetscCall(PetscFree(n_recv_displs));
  } else {
    // send the root proc the local flux metadata
    PetscCallMPI(MPI_Gatherv(local_flux_md, num_md * num_local_edges, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, rdy->comm));
  }

  PetscCall(PetscFree(local_flux_md));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitBoundaryFluxes(RDy rdy) {
  PetscFunctionBegin;

  rdy->time_series.boundary_fluxes.last_step        = -1;
  rdy->time_series.boundary_fluxes.accumulated_time = 0.0;

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
  for (PetscInt n = 0; n < num_boundary_edges; ++n) {
    rdy->time_series.boundary_fluxes.fluxes[n].current  = (TimeSeriesBoundaryFlux){0.0, 0.0, 0.0};
    rdy->time_series.boundary_fluxes.fluxes[n].previous = (TimeSeriesBoundaryFlux){0.0, 0.0, 0.0};
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateRank0VecAndVecScatter(PetscInt rank, Vec v_global, PetscInt num_sites, const PetscInt *all_sites, Vec *v_rank0,
                                                  VecScatter *scatter) {
  PetscFunctionBegin;

  PetscInt num_comp;
  PetscCall(VecGetBlockSize(v_global, &num_comp));

  // index spaces for creating the VecScatter
  IS is_from, is_to;

  if (rank == 0) {
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, num_comp * num_sites, v_rank0));
    PetscCall(VecSetBlockSize(*v_rank0, num_comp));

    // create an IS based on the natural IDs (of the observation sites) from the global vector from which data will be scattered
    PetscInt *int_array;
    PetscCall(PetscCalloc1(num_sites, &int_array));
    for (PetscInt i = 0; i < num_sites; ++i) {
      int_array[i] = all_sites[i];
    }
    PetscCall(ISCreateBlock(PETSC_COMM_SELF, num_comp, num_sites, int_array, PETSC_COPY_VALUES, &is_from));

    // create an IS for the rank 0 sequential Vec storing the selected data
    for (PetscInt i = 0; i < num_sites; ++i) {
      int_array[i] = i;
    }
    PetscCall(ISCreateBlock(PETSC_COMM_SELF, num_comp, num_sites, int_array, PETSC_COPY_VALUES, &is_to));

    PetscCall(PetscFree(int_array));
  } else {
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, 0, v_rank0));
    PetscInt *int_array;
    PetscCall(PetscCalloc1(0, &int_array));
    PetscCall(ISCreateBlock(PETSC_COMM_SELF, num_comp, 0, int_array, PETSC_COPY_VALUES, &is_from));
    PetscCall(ISCreateBlock(PETSC_COMM_SELF, num_comp, 0, int_array, PETSC_COPY_VALUES, &is_to));
    PetscCall(PetscFree(int_array));
  }

  PetscCall(VecScatterCreate(v_global, is_from, *v_rank0, is_to, scatter));

  PetscCall(ISDestroy(&is_from));
  PetscCall(ISDestroy(&is_to));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitObservations(RDy rdy) {
  PetscFunctionBegin;

  rdy->time_series.observations.last_step        = -1;
  rdy->time_series.observations.accumulated_time = 0.0;

  // create an accumulation vector for storing instantaneous or time-averaged solution data
  PetscCall(VecDuplicate(rdy->u_global, &rdy->time_series.observations.accum_u));

  // determine which sites are local to this process and record their global and local ids
  PetscInt  num_sites = rdy->config.output.time_series.observations.sites.cells_count;
  PetscInt *all_sites = rdy->config.output.time_series.observations.sites.cells;

  PetscCall(CreateRank0VecAndVecScatter(rdy->rank, rdy->u_global, num_sites, all_sites, &rdy->time_series.observations.sites.u,
                                        &rdy->time_series.observations.scatter_u));

  // set up storage on rank 0 for observations and populate the global site indices and the
  // coordinate arrays, both of which are fixed for the duration of the simulation
  {
    if (rdy->rank == 0) {
      PetscCall(PetscCalloc1(num_sites, &rdy->time_series.observations.sites.x));
      PetscCall(PetscCalloc1(num_sites, &rdy->time_series.observations.sites.y));
      PetscCall(PetscCalloc1(num_sites, &rdy->time_series.observations.sites.z));
    }

    // now get x/y/z coordinates of the sites on the rank 0
    RDyMesh  *mesh  = &rdy->mesh;
    RDyCells *cells = &mesh->cells;

    Vec        coord_rank0, coord_global;
    VecScatter scatter_1dof;

    // create a 1-DOF Vec and a Vec and VecScatter on rank 0 to exchange coordinates one at a time
    PetscCall(RDyCreateOneDOFGlobalVec(rdy, &coord_global));
    PetscCall(CreateRank0VecAndVecScatter(rdy->rank, coord_global, num_sites, all_sites, &coord_rank0, &scatter_1dof));

    // now loop over the coordinates and gather them one at a time
    for (PetscInt icoord = 0; icoord < 3; ++icoord) {
      PetscReal *coord_ptr;
      PetscCall(VecGetArray(coord_global, &coord_ptr));
      PetscInt count = 0;

      // only put coordinates of owned cells
      for (PetscInt i = 0; i < mesh->num_cells; ++i) {
        if (cells->is_owned[i]) {
          if (icoord == 0) {
            coord_ptr[count] = cells->centroids[i].X[0];
          } else if (icoord == 1) {
            coord_ptr[count] = cells->centroids[i].X[1];
          } else {
            coord_ptr[count] = cells->centroids[i].X[2];
          }
          count++;
        }
      }

      PetscCall(VecRestoreArray(coord_global, &coord_ptr));

      // scatter the coordinates to rank 0
      PetscCall(VecScatterBegin(scatter_1dof, coord_global, coord_rank0, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(scatter_1dof, coord_global, coord_rank0, INSERT_VALUES, SCATTER_FORWARD));

      // now copy the coordinates to the observation sites
      if (rdy->rank == 0) {
        PetscCall(VecGetArray(coord_rank0, &coord_ptr));
        for (PetscInt i = 0; i < num_sites; ++i) {
          if (icoord == 0) {
            rdy->time_series.observations.sites.x[i] = coord_ptr[i];
          } else if (icoord == 1) {
            rdy->time_series.observations.sites.y[i] = coord_ptr[i];
          } else {
            rdy->time_series.observations.sites.z[i] = coord_ptr[i];
          }
        }
        PetscCall(VecRestoreArray(coord_rank0, &coord_ptr));
      }
    }

    // clean up
    PetscCall(VecDestroy(&coord_global));
    PetscCall(VecDestroy(&coord_rank0));
    PetscCall(VecScatterDestroy(&scatter_1dof));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// initializes time series data storage
PetscErrorCode InitTimeSeries(RDy rdy) {
  PetscFunctionBegin;
  rdy->time_series = (RDyTimeSeriesData){0};
  if (rdy->config.output.time_series.boundary_fluxes > 0) {
    InitBoundaryFluxes(rdy);
  }
  if (rdy->config.output.time_series.observations.interval > 0) {
    InitObservations(rdy);
  }
  if (rdy->config.output.time_series.boundary_fluxes > 0 || rdy->config.output.time_series.observations.interval > 0) {
    PetscCall(TSMonitorSet(rdy->ts, WriteTimeSeries, rdy, NULL));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// records boundary fluxes on the given boundary from the given array of
// fluxes on boundary edges
static PetscErrorCode RecordBoundaryFluxes(RDy rdy, RDyBoundary boundary, OperatorData boundary_fluxes) {
  PetscFunctionBegin;
  RDyTimeSeriesData *time_series = &rdy->time_series;
  if (time_series->boundary_fluxes.fluxes) {
    // if the boundary condition for this boundary is not auto-generated,
    // record fluxes locally
    if (!rdy->boundary_conditions[boundary.index].auto_generated) {
      PetscInt n = rdy->time_series.boundary_fluxes.offsets[boundary.index];
      for (PetscInt e = 0; e < boundary.num_edges; ++e) {
        PetscInt  edge_id  = boundary.edge_ids[e];
        PetscInt  cell_id  = rdy->mesh.edges.cell_ids[2 * edge_id];
        PetscReal edge_len = rdy->mesh.edges.lengths[edge_id];

        if (rdy->mesh.cells.is_owned[cell_id]) {
          // FIXME: this is specific to the shallow water equations

          // multiply by the edge length
          time_series->boundary_fluxes.fluxes[n].current.water_mass = edge_len * boundary_fluxes.values[0][e];
          time_series->boundary_fluxes.fluxes[n].current.x_momentum = edge_len * boundary_fluxes.values[1][e];
          time_series->boundary_fluxes.fluxes[n].current.y_momentum = edge_len * boundary_fluxes.values[2][e];
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

  PetscReal accumulated_time = rdy->time_series.boundary_fluxes.accumulated_time;
  if (accumulated_time == 0.0)
    accumulated_time = 1.0;  // to avoid division by zero in case this is our first step or if the time interval between steps is zero for some reason

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
          local_flux_data[num_data * n]     = rdy->time_series.boundary_fluxes.fluxes[n].current.water_mass / accumulated_time;
          local_flux_data[num_data * n + 1] = rdy->time_series.boundary_fluxes.fluxes[n].current.x_momentum / accumulated_time;
          local_flux_data[num_data * n + 2] = rdy->time_series.boundary_fluxes.fluxes[n].current.y_momentum / accumulated_time;
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
    PetscCallMPI(MPI_Gatherv(local_flux_data, num_data * num_local_edges, MPI_DOUBLE, global_flux_data, n_recv_counts, n_recv_displs, MPI_DOUBLE, 0,
                             rdy->comm));
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
    PetscCallMPI(MPI_Gatherv(local_flux_data, num_data * num_local_edges, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, rdy->comm));
  }
  PetscCall(PetscFree(local_flux_data));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode WriteObservations(RDy rdy, PetscInt step, PetscReal time) {
  PetscFunctionBegin;

  // scatter the accumulation vector to the sites vector on rank 0
  PetscCall(VecScatterBegin(rdy->time_series.observations.scatter_u, rdy->time_series.observations.accum_u, rdy->time_series.observations.sites.u,
                            INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(rdy->time_series.observations.scatter_u, rdy->time_series.observations.accum_u, rdy->time_series.observations.sites.u,
                          INSERT_VALUES, SCATTER_FORWARD));

  if (rdy->rank == 0) {
    // open the file in the appropriate writing mode
    char output_dir[PETSC_MAX_PATH_LEN], prefix[PETSC_MAX_PATH_LEN], path[PETSC_MAX_PATH_LEN];
    PetscCall(GetOutputDirectory(rdy, output_dir));
    PetscCall(DetermineConfigPrefix(rdy, prefix));
    snprintf(path, PETSC_MAX_PATH_LEN - 1, "%s/%s-observations.dat", output_dir, prefix);

    FILE *fp = NULL;
    if (step == 0) {  // write a header on the first step
      PetscCall(PetscFOpen(rdy->comm, path, "w", &fp));
      PetscCall(PetscFPrintf(rdy->comm, fp, "# time      \tx        \ty       \tz          \tname       \tvalue\n"));
    } else {
      PetscCall(PetscFOpen(rdy->comm, path, "a", &fp));
    }

    PetscInt num_sites, num_comp;
    PetscCall(VecGetBlockSize(rdy->time_series.observations.sites.u, &num_comp));
    PetscCall(VecGetSize(rdy->time_series.observations.sites.u, &num_sites));
    num_sites /= num_comp;

    // retrieve the names of the solution vector components
    const char  *component_names[MAX_NUM_FIELD_COMPONENTS];
    PetscSection section;
    PetscCall(DMGetLocalSection(rdy->dm, &section));
    for (PetscInt c = 0; c < num_comp; ++c) {
      PetscCall(PetscSectionGetComponentName(section, 0, c, &component_names[c]));
    }

    const PetscReal *values;
    PetscCall(VecGetArrayRead(rdy->time_series.observations.sites.u, &values));
    for (PetscInt i = 0; i < num_sites; ++i) {
      PetscReal x = rdy->time_series.observations.sites.x[i];
      PetscReal y = rdy->time_series.observations.sites.y[i];
      PetscReal z = rdy->time_series.observations.sites.z[i];
      for (PetscInt c = 0; c < num_comp; ++c) {
        PetscReal value = values[num_comp * i + c];
        PetscCall(PetscFPrintf(rdy->comm, fp, "%e\t%f\t%f\t%f\t%s  \t%e\n", time, x, y, z, component_names[c], value));
      }
    }
    PetscCall(VecRestoreArrayRead(rdy->time_series.observations.sites.u, &values));
    PetscCall(PetscFClose(rdy->comm, fp));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrintCeedVector(CeedVector vec) {
  PetscFunctionBegin;
  CeedSize length;
  PetscCallCEED(CeedVectorGetLength(vec, &length));
  PetscReal *array;
  PetscCallCEED(CeedVectorGetArray(vec, CEED_MEM_HOST, &array));
  for (CeedInt i = 0; i < length / 3; ++i) {
    PetscPrintf(PETSC_COMM_SELF, "%d %f\n", i, array[i * 3]);
  }
  PetscPrintf(PETSC_COMM_SELF, "\n");
  PetscCallCEED(CeedVectorRestoreArray(vec, &array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AccumulateBoundaryFluxes(RDy rdy) {
  PetscFunctionBegin;

  PetscReal dt = rdy->dt;
  rdy->time_series.boundary_fluxes.accumulated_time += dt;

  if (CeedEnabled()) {
    Operator *op = rdy->operator;

    for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
      RDyBoundary *boundary = &rdy->boundaries[b];

      // get the relevant boundary sub-operator
      CeedOperator *sub_ops;
      PetscCallCEED(CeedOperatorCompositeGetSubList(op->ceed.flux, &sub_ops));
      CeedOperator sub_op = sub_ops[1 + boundary->index];

      // fetch the relevant vector
      CeedOperatorField field;
      PetscCallCEED(CeedOperatorGetFieldByName(sub_op, "flux", &field));
      CeedVector vec;
      PetscCallCEED(CeedOperatorFieldGetVector(field, &vec));

      if (0) {
        printf("\nBefore accumulation for boundary %d:\n", (int)boundary->index);
        PetscCall(PrintCeedVector(boundary->flux_accumulated));
      }
      PetscCallCEED(CeedVectorAXPY(boundary->flux_accumulated, dt, vec));
      if (0) {
        printf("After accumulation for boundary %d:\n", (int)boundary->index);
        PetscCall(PrintCeedVector(boundary->flux_accumulated));
      }
    }
  } else {
    // For PETSc, the ApplyBoundaryFlux does the accumulation internally, so we don't need to do anything here.
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Resets accumulated boundary flux data to zero after writing time series output.
// This clears both the accumulated time counter and the CEED flux accumulation vectors.
static PetscErrorCode ResetAccumulatedBoundaryFluxes(RDy rdy) {
  PetscFunctionBegin;

  rdy->time_series.boundary_fluxes.accumulated_time = 0.0;

  if (CeedEnabled()) {
    for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
      RDyBoundary *boundary = &rdy->boundaries[b];
      CeedVector   vec      = boundary->flux_accumulated;
      PetscCallCEED(CeedVectorSetValue(vec, 0.0));
    }
  } else {
    Operator *op = rdy->operator;
    for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
      PetscCall(VecZeroEntries(op->petsc.boundary_fluxes_accum[b]));
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// This monitoring function writes out all requested time series data.
PetscErrorCode WriteTimeSeries(TS ts, PetscInt step, PetscReal time, Vec X, void *ctx) {
  PetscFunctionBegin;

  RDy rdy = ctx;

  // observations
  int observations_interval = rdy->config.output.time_series.observations.interval;
  if (observations_interval && (step % observations_interval == 0) && (step > rdy->time_series.observations.last_step)) {
    // FIXME: This VecCopy is only appropriate for instantaneous observations; we haven't figured out
    // FIXME: how to articulate the time averaged values (and their windows) yet. We should revisit this
    // FIXME: when we want averaged values.
    PetscCall(VecCopy(rdy->u_global, rdy->time_series.observations.accum_u));
    PetscCall(WriteObservations(rdy, step, time));
  }

  // boundary fluxes
  PetscCall(AccumulateBoundaryFluxes(rdy));

  int boundary_flux_interval = rdy->config.output.time_series.boundary_fluxes;
  if ((step % boundary_flux_interval == 0) && (step > rdy->time_series.boundary_fluxes.last_step)) {
    for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
      RDyBoundary *boundary = &rdy->boundaries[b];

      OperatorData boundary_fluxes;
      PetscCall(ExtractOperatorBoundaryFluxes(rdy->operator, boundary, &boundary_fluxes));
      PetscCall(RecordBoundaryFluxes(rdy, *boundary, boundary_fluxes));
      PetscCall(DestroyOperatorData(&boundary_fluxes));
    }
    PetscCall(WriteBoundaryFluxes(rdy, step, time));
    PetscCall(ResetAccumulatedBoundaryFluxes(rdy));
    rdy->time_series.boundary_fluxes.last_step = step;
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
  if (rdy->time_series.observations.sites.x) {
    PetscFree(rdy->time_series.observations.sites.x);
    PetscFree(rdy->time_series.observations.sites.y);
    PetscFree(rdy->time_series.observations.sites.z);
    PetscCall(VecDestroy(&rdy->time_series.observations.sites.u));
    PetscCall(VecScatterDestroy(&rdy->time_series.observations.scatter_u));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
