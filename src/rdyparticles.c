// rdyparticles.c — Lagrangian particle tracer implementation for RDycore
//
// Particles are advected as an inline diagnostic using PETSc's DMSwarm with
// DMSWARM_PIC type. The design follows the pattern from PETSc ex77.c:
//   - A separate TS for particle advection with a FreeStreaming RHS
//   - DMSwarmCreateGlobalVectorFromField wraps particle coordinates as a Vec
//   - TSSetPostStep on the main PDE TS triggers AdvectParticles each step
//   - DMSwarmMigrate handles particles crossing MPI partition boundaries
//
// Velocity is computed from SWE conserved variables (h, hu, hv) at cell
// centers using P0 (cell-constant) interpolation. Phase 6 will add P1 vertex
// projection via DMInterpolationInfo.
//
// Enable via: -particles_per_cell N  (N > 0 enables, N = 0 disables)

#include <petscdmswarm.h>
#include <petscviewerhdf5.h>  // note: includes hdf5.h
#include <petscts.h>
#include <private/rdycoreimpl.h>
#include <private/rdymeshimpl.h>
#include <private/rdyparticlesimpl.h>
#include <string.h>

// ---------------------------------------------------------------------------
// FreeStreaming — RHS function for the particle TS
//
// Signature required by TSSetRHSFunction:
//   PetscErrorCode f(TS ts, PetscReal t, Vec X, Vec F, void *ctx)
//
// X holds particle coordinates (2*Np scalars: x0,y0, x1,y1, ...).
// F is filled with the velocity at each particle position (dx/dt = v).
//
// For Phase 1-5 we use cell-constant (P0) velocity: look up the cell that
// owns each particle via DMSwarmPICField_cellid and read hu/h, hv/h from
// the PDE solution vector.
// ---------------------------------------------------------------------------
static PetscErrorCode FreeStreaming(TS ts, PetscReal t, Vec X, Vec F, void *ctx) {
  PetscFunctionBegin;
  (void)t;
  (void)X;  // X unused for P0 cell-constant velocity; needed in Phase 6a (P1 interpolation)

  ParticleAdvCtx *adv = (ParticleAdvCtx *)ctx;
  RDy             rdy = adv->rdy;
  DM              sdm;
  PetscCall(TSGetDM(ts, &sdm));

  // ---- 1. Get the number of local particles --------------------------------
  PetscInt Np;
  PetscCall(DMSwarmGetLocalSize(sdm, &Np));

  // ---- 2. Access the PDE solution (array already pinned by AdvectParticles) -
  // u_global stores [h, hu, hv, ...] per cell (3 + num_tracers components).
  PetscInt ndof;
  PetscCall(VecGetBlockSize(adv->u_pde, &ndof));

  // u_arr is pinned for the duration of TSSolve by AdvectParticles.
  // We call VecGetArrayRead here too — PETSc ref-counts the pin, so this is
  // cheap (no copy) and safe to pair with a matching VecRestoreArrayRead below.
  const PetscScalar *u_arr;
  PetscCall(VecGetArrayRead(adv->u_pde, &u_arr));

  // ---- 3. Use cached cell ID field name (set once in InitParticleTS) ------
  PetscInt *cell_ids;
  PetscCall(DMSwarmGetField(sdm, adv->cellid_field, NULL, NULL, (void **)&cell_ids));

  // ---- 4. Fill F with velocity at each particle position ------------------
  PetscScalar *f_arr;
  PetscCall(VecGetArray(F, &f_arr));

  PetscInt num_local_cells = rdy->mesh.num_owned_cells;

  for (PetscInt p = 0; p < Np; ++p) {
    PetscInt  cid   = cell_ids[p];
    PetscReal vel_x = 0.0, vel_y = 0.0;

    // Only compute velocity for valid local cells
    if (cid >= 0 && cid < num_local_cells) {
      PetscReal h  = (PetscReal)u_arr[cid * ndof + 0];
      PetscReal hu = (PetscReal)u_arr[cid * ndof + 1];
      PetscReal hv = (PetscReal)u_arr[cid * ndof + 2];
      if (h > adv->tiny_h) {
        vel_x = hu / h;
        vel_y = hv / h;
      }
    }

    f_arr[p * 2 + 0] = vel_x;
    f_arr[p * 2 + 1] = vel_y;
  }

  PetscCall(VecRestoreArray(F, &f_arr));
  PetscCall(DMSwarmRestoreField(sdm, adv->cellid_field, NULL, NULL, (void **)&cell_ids));
  PetscCall(VecRestoreArrayRead(adv->u_pde, &u_arr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
// AdvectParticles — TSPostStep callback registered on the main PDE TS
//
// Called once per accepted PDE time step. Follows ex77.c lines 834-871:
//   1. Retrieve the particle TS via PetscObjectQuery
//   2. Wrap particle coordinates as a Vec
//   3. Set particle TS time window to match the PDE step
//   4. TSSolve the particle ODE: dx/dt = v(x)
//   5. Return the coordinate Vec to the swarm
//   6. DMSwarmMigrate to relocate particles across MPI ranks
//   7. TSReset if particle counts changed
// ---------------------------------------------------------------------------
static PetscErrorCode AdvectParticles(TS ts) {
  PetscFunctionBegin;

  // Retrieve the particle TS attached to the main TS
  TS sts = NULL;
  PetscCall(PetscObjectQuery((PetscObject)ts, "_SwarmTS", (PetscObject *)&sts));
  if (!sts) PetscFunctionReturn(PETSC_SUCCESS);  // particles not enabled

  // Get the DMSwarm and advection context from the particle TS
  DM              sdm;
  ParticleAdvCtx *adv;
  PetscCall(TSGetDM(sts, &sdm));
  PetscCall(TSGetApplicationContext(sts, (void **)&adv));

  // Record current local particle count for migration check (local-only, no MPI_Allreduce)
  PetscInt Np_before;
  PetscCall(DMSwarmGetLocalSize(sdm, &Np_before));

  // Get current PDE time.
  // Following ex77.c lines 852-855: the particle TS maintains its own time
  // state starting from 0; we only set TSSetMaxTime to the current PDE time
  // and let the particle TS advance from wherever it left off.
  PetscReal time;
  PetscCall(TSGetTime(ts, &time));

  // Wrap particle coordinates as a Vec for TSSolve
  Vec coordinates;
  PetscCall(DMSwarmCreateGlobalVectorFromField(sdm, DMSwarmPICField_coor, &coordinates));

  // Set the target time for the particle TS (ex77.c line 853).
  // Do NOT call TSSetTime or TSSetTimeStep — the particle TS manages its own
  // internal time and step size.
  PetscCall(TSSetMaxTime(sts, time));

  PetscCall(TSSetExactFinalTime(sts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTimeStep(sts, adv->rdy->dt));

  // Solve the particle ODE: dx/dt = v(x)
  PetscPreLoadBegin(PETSC_FALSE, "RDyParticle solve");
  PetscCall(TSSolve(sts, coordinates));
  PetscPreLoadEnd();

  // Return the coordinate Vec to the swarm
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sdm, DMSwarmPICField_coor, &coordinates));

  // Migrate particles to correct MPI ranks (ex77.c line 860).
  // Must happen BEFORE velocity field population so that cell IDs are valid
  // for the local rank's owned cells.
  PetscPreLoadBegin(PETSC_FALSE, "RDyParticles: DMSwarmMigrate");
  PetscCall(DMSwarmMigrate(sdm, PETSC_TRUE));
  PetscPreLoadEnd();

  // Reset particle TS if local counts changed after migration (ex77.c lines 863-868).
  PetscPreLoadBegin(PETSC_FALSE, "RDyParticles: TSReset");
  PetscInt Np_after;
  PetscCall(DMSwarmGetLocalSize(sdm, &Np_after));
  if (Np_before != Np_after) {
    PetscInt Np_global_before = 0, Np_global_after = 0;
    MPI_Allreduce(&Np_before, &Np_global_before, 1, MPIU_INT, MPI_SUM, adv->rdy->comm);
    MPI_Allreduce(&Np_after, &Np_global_after, 1, MPIU_INT, MPI_SUM, adv->rdy->comm);
    RDyLogDebug(adv->rdy,
                "AdvectParticles: particle count changed after DMSwarmMigrate "
                "(global: %" PetscInt_FMT " -> %" PetscInt_FMT ", lost %" PetscInt_FMT ")",
                Np_global_before, Np_global_after, Np_global_before - Np_global_after);
    PetscCall(TSReset(sts));
    PetscCall(DMSwarmVectorDefineField(sdm, DMSwarmPICField_coor));
  }
  PetscPreLoadEnd();

  // ---- Populate velocity fields for output (once per step, not per RHS call) ----
  // Done after migration so cell IDs correspond to local owned cells.
  {
    PetscInt Np;
    PetscCall(DMSwarmGetLocalSize(sdm, &Np));

    PetscInt ndof;
    PetscCall(VecGetBlockSize(adv->u_pde, &ndof));

    const PetscScalar *u_arr;
    PetscCall(VecGetArrayRead(adv->u_pde, &u_arr));

    PetscInt *cell_ids;
    PetscCall(DMSwarmGetField(sdm, adv->cellid_field, NULL, NULL, (void **)&cell_ids));

    PetscReal *vx_field, *vy_field;
    PetscCall(DMSwarmGetField(sdm, "velocity_x", NULL, NULL, (void **)&vx_field));
    PetscCall(DMSwarmGetField(sdm, "velocity_y", NULL, NULL, (void **)&vy_field));

    PetscInt num_local_cells = adv->rdy->mesh.num_owned_cells;
    for (PetscInt p = 0; p < Np; ++p) {
      PetscInt  cid   = cell_ids[p];
      PetscReal vel_x = 0.0, vel_y = 0.0;
      if (cid >= 0 && cid < num_local_cells) {
        PetscReal h  = (PetscReal)u_arr[cid * ndof + 0];
        PetscReal hu = (PetscReal)u_arr[cid * ndof + 1];
        PetscReal hv = (PetscReal)u_arr[cid * ndof + 2];
        if (h > adv->tiny_h) {
          vel_x = hu / h;
          vel_y = hv / h;
        }
      }
      vx_field[p] = vel_x;
      vy_field[p] = vel_y;
    }

    PetscCall(DMSwarmRestoreField(sdm, "velocity_y", NULL, NULL, (void **)&vy_field));
    PetscCall(DMSwarmRestoreField(sdm, "velocity_x", NULL, NULL, (void **)&vx_field));
    PetscCall(DMSwarmRestoreField(sdm, adv->cellid_field, NULL, NULL, (void **)&cell_ids));
    PetscCall(VecRestoreArrayRead(adv->u_pde, &u_arr));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
// InitParticleSwarmDM — Create and seed the DMSwarm
//
// Follows ex77.c lines 893-939 (PART_LAYOUT_CELL pattern).
// ---------------------------------------------------------------------------
static PetscErrorCode InitParticleSwarmDM(RDy rdy) {
  PetscFunctionBegin;

  RDyParticles *parts = &rdy->particles;
  MPI_Comm      comm  = rdy->comm;
  PetscInt      Npc   = parts->particles_per_cell;

  RDyLogDebug(rdy, "Creating particle swarm DM with %d particle(s) per cell...", (int)Npc);

  // ---- Create the DMSwarm -------------------------------------------------
  DM sdm;
  PetscCall(DMCreate(comm, &sdm));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)sdm, "part_"));
  PetscCall(DMSetType(sdm, DMSWARM));
  PetscCall(DMSetDimension(sdm, 3));

  // Attach the primary DMPlex as the cell DM so DMSwarm knows the spatial
  // decomposition and can locate particles in cells.
  PetscCall(DMSwarmSetCellDM(sdm, rdy->dm));
  PetscCall(DMSwarmSetType(sdm, DMSWARM_PIC));

  // ---- Register custom fields for velocity storage ------------------------
  PetscCall(DMSwarmRegisterPetscDatatypeField(sdm, "velocity_x", 1, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(sdm, "velocity_y", 1, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(sdm));

  // ---- Seed particles — one per owned cell (Npc == 1: cell centroid) -----
  // We do NOT use DMSwarmSetPointCoordinatesRandom because it iterates over
  // ALL local cells from DMPlexGetHeightStratum (including ghost cells on
  // distributed meshes), which writes past the coordinate array when the
  // swarm is sized for owned cells only. Instead, we seed particles at cell
  // centroids for owned cells only.
  PetscInt num_owned_cells = rdy->mesh.num_owned_cells;
  PetscInt swarm_buffer    = PetscMax(num_owned_cells * Npc / 4, 8);
  PetscCall(DMSwarmSetLocalSizes(sdm, num_owned_cells * Npc, swarm_buffer));

  {
    PetscInt  cStart, cEnd, dim;
    PetscReal centroid[3];
    PetscCall(DMGetDimension(rdy->dm, &dim));
    PetscCall(DMPlexGetHeightStratum(rdy->dm, 0, &cStart, &cEnd));

    PetscReal *coords;
    PetscCall(DMSwarmGetField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));

    PetscInt p = 0;  // particle index (only owned cells)
    for (PetscInt c = cStart; c < cEnd; ++c) {
      // Skip ghost cells: check if the cell has a non-negative global offset
      PetscInt gref, junk;
      PetscCall(DMPlexGetPointGlobal(rdy->dm, c, &gref, &junk));
      if (gref < 0) continue;  // ghost cell

      PetscCall(DMPlexComputeCellGeometryFVM(rdy->dm, c, NULL, centroid, NULL));
      for (PetscInt d = 0; d < dim; ++d) {
        coords[p * dim + d] = centroid[d];
      }
      ++p;
    }
    PetscCheck(p == num_owned_cells * Npc, comm, PETSC_ERR_PLIB,
               "Particle count mismatch: seeded %" PetscInt_FMT " but expected %" PetscInt_FMT,
               p, num_owned_cells * Npc);

    PetscCall(DMSwarmRestoreField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  }

  // ---- Define the coordinate field as the Vec field for the particle TS --
  // (ex77.c line 939)
  PetscCall(DMSwarmVectorDefineField(sdm, DMSwarmPICField_coor));

  parts->dm_swarm = sdm;

  PetscInt global_np;
  PetscCall(DMSwarmGetSize(sdm, &global_np));
  RDyLogDebug(rdy, "Particle swarm created: %d global particles", (int)global_np);

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
// InitParticleTS — Create the separate TS for particle advection
//
// Follows ex77.c lines 921-938.
// ---------------------------------------------------------------------------
static PetscErrorCode InitParticleTS(RDy rdy) {
  PetscFunctionBegin;

  RDyParticles *parts = &rdy->particles;
  MPI_Comm      comm  = rdy->comm;

  // ---- Set up the advection context ---------------------------------------
  parts->adv_ctx.u_pde  = rdy->u_global;
  parts->adv_ctx.tiny_h = rdy->config.physics.flow.tiny_h;
  parts->adv_ctx.rdy    = rdy;

  // Cache the cell ID field name — constant for the swarm lifetime, avoids
  // calling DMSwarmGetCellDMActive + DMSwarmCellDMGetCellID on every RHS call.
  {
    DMSwarmCellDM celldm;
    PetscCall(DMSwarmGetCellDMActive(parts->dm_swarm, &celldm));
    PetscCall(DMSwarmCellDMGetCellID(celldm, &parts->adv_ctx.cellid_field));
  }

  // ---- Create the particle TS ---------------------------------------------
  TS sts;
  PetscCall(TSCreate(comm, &sts));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)sts, "part_"));
  PetscCall(TSSetDM(sts, parts->dm_swarm));
  PetscCall(TSSetProblemType(sts, TS_NONLINEAR));
  PetscCall(TSSetExactFinalTime(sts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetRHSFunction(sts, NULL, FreeStreaming, &parts->adv_ctx));

  // Store adv_ctx as application context so AdvectParticles can retrieve it
  // via TSGetApplicationContext without needing a non-existent API.
  PetscCall(TSSetApplicationContext(sts, &parts->adv_ctx));

  // Default to forward Euler; user can override with -part_ts_type
  PetscCall(TSSetType(sts, TSEULER));
  PetscCall(TSSetFromOptions(sts));

  parts->ts_particles = sts;

  // ---- Attach particle TS to main TS via PetscObjectCompose ---------------
  // (ex77.c line 938) — AdvectParticles retrieves it via PetscObjectQuery
  PetscCall(PetscObjectCompose((PetscObject)rdy->ts, "_SwarmTS", (PetscObject)sts));

  // ---- Register AdvectParticles as PostStep on the main PDE TS -----------
  // (ex77.c line 937)
  PetscCall(TSSetPostStep(rdy->ts, AdvectParticles));

  RDyLogDebug(rdy, "Particle TS created and PostStep registered.");

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Creates the particle swarm DM and its advection TS.
/// Called from RDySetup after InitSolver.
/// Reads -particles_per_cell from options; does nothing if N <= 0.
PetscErrorCode CreateParticleSwarm(RDy rdy) {
  PetscFunctionBegin;

  // Read the number of particles per cell from command-line options
  rdy->particles.particles_per_cell = 0;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-particles_per_cell", &rdy->particles.particles_per_cell, NULL));

  if (rdy->particles.particles_per_cell <= 0) {
    rdy->particles.enabled = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  rdy->particles.enabled = PETSC_TRUE;

  PetscCall(InitParticleSwarmDM(rdy));
  PetscCall(InitParticleTS(rdy));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Destroys the particle swarm and its advection TS.
/// Called from RDyDestroy.
PetscErrorCode DestroyParticleSwarm(RDy rdy) {
  PetscFunctionBegin;

  if (!rdy->particles.enabled) PetscFunctionReturn(PETSC_SUCCESS);

  if (rdy->particles.ts_particles) {
    PetscCall(TSDestroy(&rdy->particles.ts_particles));
  }
  if (rdy->particles.dm_swarm) {
    PetscCall(DMDestroy(&rdy->particles.dm_swarm));
  }
  rdy->particles.enabled = PETSC_FALSE;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Writes particle positions to HDF5 + XDMF files compatible with VisIT/ParaView.
/// Each call produces:
///   <output_dir>/<prefix>-particles-<step06d>.h5
///   <output_dir>/<prefix>-particles-<step06d>.xmf
PetscErrorCode WriteParticleOutput(RDy rdy, PetscInt step, PetscReal time) {
  PetscFunctionBegin;

  if (!rdy->particles.enabled) PetscFunctionReturn(PETSC_SUCCESS);

  DM sdm = rdy->particles.dm_swarm;

  // ---- 1. Get local and global particle counts ----------------------------
  PetscInt Np_local;
  PetscCall(DMSwarmGetLocalSize(sdm, &Np_local));

  MPI_Comm comm = rdy->comm;

  // per-rank write offset via exclusive scan
  PetscInt rank_offset = 0;
  MPI_Exscan(&Np_local, &rank_offset, 1, MPIU_INT, MPI_SUM, comm);

  PetscInt Np_global = 0;
  MPI_Allreduce(&Np_local, &Np_global, 1, MPIU_INT, MPI_SUM, comm);

  // ---- 2. Build filenames -------------------------------------------------
  char output_dir[PETSC_MAX_PATH_LEN];
  PetscCall(GetOutputDirectory(rdy, output_dir));
  char prefix[PETSC_MAX_PATH_LEN];
  PetscCall(DetermineConfigPrefix(rdy, prefix));

  char h5_path[PETSC_MAX_PATH_LEN], xmf_path[PETSC_MAX_PATH_LEN];
  snprintf(h5_path, PETSC_MAX_PATH_LEN, "%s/%s-particles-%06" PetscInt_FMT ".h5", output_dir, prefix, step);
  snprintf(xmf_path, PETSC_MAX_PATH_LEN, "%s/%s-particles-%06" PetscInt_FMT ".xmf", output_dir, prefix, step);

  // basename for XMF reference so the two files can be moved together
  const char *h5_basename = strrchr(h5_path, '/');
  h5_basename             = h5_basename ? h5_basename + 1 : h5_path;

  // ---- 3. Write HDF5 with collective MPI-IO -------------------------------
  hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL);
  hid_t file_id = H5Fcreate(h5_path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
  H5Pclose(fapl);

  // Create /Coordinates dataset: global shape (Np_global, 3)
  hsize_t gdims[2] = {(hsize_t)Np_global, 3};
  hid_t   fspace   = H5Screate_simple(2, gdims, NULL);
  hid_t   dset     = H5Dcreate2(file_id, "/Coordinates", H5T_NATIVE_DOUBLE, fspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Collective transfer property
  hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);

  // Fetch raw 2-D coordinates (x, y) from DMSwarm; pad z=0 for XDMF XYZ
  PetscReal *raw_coords;
  PetscCall(DMSwarmGetField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **)&raw_coords));

  double *buf = NULL;
  if (Np_local > 0) {
    PetscCall(PetscMalloc1(Np_local * 3, &buf));
    for (PetscInt p = 0; p < Np_local; ++p) {
      buf[p * 3 + 0] = (double)raw_coords[p * 2 + 0];
      buf[p * 3 + 1] = (double)raw_coords[p * 2 + 1];
      buf[p * 3 + 2] = 0.0;
    }
  }

  PetscCall(DMSwarmRestoreField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **)&raw_coords));

  // File hyperslab: each rank writes its contiguous block
  if (Np_local > 0) {
    hsize_t offset[2] = {(hsize_t)rank_offset, 0};
    hsize_t count[2]  = {(hsize_t)Np_local, 3};
    H5Sselect_hyperslab(fspace, H5S_SELECT_SET, offset, NULL, count, NULL);
  } else {
    // Ranks with no particles must still participate in collective I/O
    H5Sselect_none(fspace);
  }

  // Memory dataspace
  hsize_t mem_dims[2] = {(hsize_t)Np_local, 3};
  hid_t   mspace     = (Np_local > 0) ? H5Screate_simple(2, mem_dims, NULL) : H5Screate(H5S_NULL);

  H5Dwrite(dset, H5T_NATIVE_DOUBLE, mspace, fspace, dxpl, buf);

  H5Sclose(mspace);
  H5Sclose(fspace);
  H5Pclose(dxpl);
  H5Dclose(dset);
  H5Fclose(file_id);

  if (buf) PetscCall(PetscFree(buf));

  // ---- 4. Write XMF (rank 0 only) -----------------------------------------
  PetscMPIInt rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == 0) {
    FILE *fp;
    PetscCall(PetscFOpen(PETSC_COMM_SELF, xmf_path, "w", &fp));
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp,
                           "<?xml version=\"1.0\" ?>\n"
                           "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n"
                           "<Xdmf Version=\"3.0\">\n"
                           "  <Domain>\n"
                           "    <Grid Name=\"Particles\" GridType=\"Uniform\">\n"
                           "      <Topology TopologyType=\"Polyvertex\" Dimensions=\"%" PetscInt_FMT "\"/>\n"
                           "      <Geometry GeometryType=\"XYZ\">\n"
                           "        <DataItem Format=\"HDF\" Dimensions=\"%" PetscInt_FMT " 3\" NumberType=\"Float\" Precision=\"8\">\n"
                           "          %s:/Coordinates\n"
                           "        </DataItem>\n"
                           "      </Geometry>\n"
                           "    </Grid>\n"
                           "  </Domain>\n"
                           "</Xdmf>\n",
                           Np_global, Np_global, h5_basename));
    PetscCall(PetscFClose(PETSC_COMM_SELF, fp));
  }

  RDyLogDebug(rdy, "Wrote particle output: %s (step=%" PetscInt_FMT ", t=%g)", xmf_path, step, (double)time);

  PetscFunctionReturn(PETSC_SUCCESS);
}
