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
#include <petscts.h>
#include <private/rdycoreimpl.h>
#include <private/rdymeshimpl.h>
#include <private/rdyparticlesimpl.h>

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

  // Solve the particle ODE: dx/dt = v(x)
  PetscCall(TSSolve(sts, coordinates));

  // Return the coordinate Vec to the swarm
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sdm, DMSwarmPICField_coor, &coordinates));

  // Migrate particles to correct MPI ranks (ex77.c line 860).
  // Must happen BEFORE velocity field population so that cell IDs are valid
  // for the local rank's owned cells.
  PetscCall(DMSwarmMigrate(sdm, PETSC_TRUE));

  // Reset particle TS if local counts changed after migration (ex77.c lines 863-868).
  PetscInt Np_after;
  PetscCall(DMSwarmGetLocalSize(sdm, &Np_after));
  if (Np_before != Np_after) {
    PetscCall(TSReset(sts));
    PetscCall(DMSwarmVectorDefineField(sdm, DMSwarmPICField_coor));
  }

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
  PetscCall(DMSetDimension(sdm, 2));

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

/// Writes particle positions and velocities to an XDMF file.
/// Called from WriteXDMFOutput when particles are enabled.
/// Generates filename: <output_dir>/<prefix>-particles-<step>.xmf
PetscErrorCode WriteParticleOutput(RDy rdy, PetscInt step, PetscReal time) {
  PetscFunctionBegin;

  if (!rdy->particles.enabled) PetscFunctionReturn(PETSC_SUCCESS);

  DM sdm = rdy->particles.dm_swarm;

  // Build the output filename
  char output_dir[PETSC_MAX_PATH_LEN];
  PetscCall(GetOutputDirectory(rdy, output_dir));

  char prefix[PETSC_MAX_PATH_LEN];
  PetscCall(DetermineConfigPrefix(rdy, prefix));

  char filename[PETSC_MAX_PATH_LEN];
  snprintf(filename, PETSC_MAX_PATH_LEN, "%s/%s-particles-%06" PetscInt_FMT ".xmf", output_dir, prefix, step);

  // Write particle fields to XDMF/HDF5
  // DMSwarmViewFieldsXDMF writes coordinates plus the listed fields
  const char *field_names[] = {"velocity_x", "velocity_y"};
  PetscCall(DMSwarmViewFieldsXDMF(sdm, filename, 2, field_names));

  RDyLogDebug(rdy, "Wrote particle output: %s (step=%" PetscInt_FMT ", t=%g)", filename, step, (double)time);

  PetscFunctionReturn(PETSC_SUCCESS);
}
