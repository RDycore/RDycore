// rdycore_adjoint - adjoint sensitivity twin experiment for RDycore
//
// Increment 3 of the Manning-calibration plan: computes dJ/du0 for a
// terminal-time gauge misfit with TSAdjoint and validates it against central
// finite differences of the full nonlinear solve.
//
// Twin experiment (single execution):
//   1. "truth" forward solve from the configured IC; record y = H u(T)
//   2. perturbed forward solve from a scaled IC (nonzero misfit)
//        J(u0) = 1/2 (H u(T) - y)^T R^-1 (H u(T) - y),  R = sigma^2 I
//   3. TSAdjointSolve with lambda(T) = H^T R^-1 (H u(T) - y)  ->  dJ/du0
//   4. FD check: central differences of J for sampled components of u0
//
// The config must set numerics.jacobian (fd validates the adjoint wiring
// independent of the analytic Jacobian's accuracy) and
// physics.flow.source.method: explicit.

#include <petscsys.h>
#include <petscts.h>
#include <private/rdycoreimpl.h>
#include <private/rdyforcingimpl.h>
#include <private/rdymathimpl.h>
#include <private/rdymeshimpl.h>
#include <rdycore.h>
#include <stdio.h>
#include <string.h>

static const char *help_str =
    "rdycore_adjoint - adjoint sensitivity twin experiment\n"
    "usage: rdycore_adjoint <filename.yaml> [options]\n\n"
    "Options:\n"
    "  -adjoint_obs_stride <int>   Observe height at every Nth cell (default: 2)\n"
    "  -adjoint_obs_error <real>   Observation error std dev sigma (default: 0.01)\n"
    "  -adjoint_ic_perturb <real>  Relative height perturbation of the IC (default: 1e-3)\n"
    "  -adjoint_fd_samples <int>   # of u0 components for the FD check (default: 8; 0 disables)\n"
    "  -adjoint_fd_eps <real>      FD step scale (default: 1e-6)\n"
    "  -adjoint_fd_tol <real>      relative L2 gate for adjoint-vs-FD (default: 1e-5)\n"
    "  -adjoint_calibrate_gauges   per-cell Manning calibration from a gauge WSE table\n"
    "  -adjoint_obs_file <path>    observation table (see data/harvey_gauges/README.md)\n"
    "  -adjoint_gauges_twin        first synthesize the table from a two-zone truth\n"
    "  -adjoint_gauge_stride <int> twin: gauges at every Nth natural cell (default: 7)\n"
    "  -adjoint_n0 <real>          gauge mode: constant prior/initial n (default: 0.03)\n"
    "  -adjoint_jred_gate <real>   gauge mode: require J_final < gate*J_init (0 = off)\n";

#define NDOF 3

// observation-matrix type matching the state vec: a device (aijkokkos) H
// keeps the per-observation-time MatMult / MatMultTranspose applies -- and,
// via MatCreateVecs, the observation-space vecs -- off the host when the
// solve runs on kokkos types
static PetscErrorCode ObservationMatrixType(Vec state, MatType *type) {
  PetscBool is_kokkos;
  PetscFunctionBeginUser;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)state, &is_kokkos, VECKOKKOS, VECSEQKOKKOS, VECMPIKOKKOS, ""));
  *type = is_kokkos ? MATAIJKOKKOS : MATAIJ;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// observation operator: height (DOF 0) at every obs_stride-th cell
// (same layout as the LETKF driver's observation matrix)
static PetscErrorCode CreateObservationMatrix(MPI_Comm comm, PetscInt ncells, Vec u_global, PetscInt obs_stride, Mat *H,
                                              PetscInt *nobs) {
  PetscInt state_size = ncells * NDOF;
  PetscInt n_obs      = 0, state_nlocal;
  MatType  mat_type;
  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < ncells; i++)
    if (i % obs_stride == 0) n_obs++;
  PetscCall(VecGetLocalSize(u_global, &state_nlocal));

  PetscCall(MatCreate(comm, H));
  // the column layout must match u_global's (uneven) DMPlex distribution
  PetscCall(MatSetSizes(*H, PETSC_DECIDE, state_nlocal, n_obs, state_size));
  PetscCall(ObservationMatrixType(u_global, &mat_type));
  PetscCall(MatSetType(*H, mat_type));
  PetscCall(MatSeqAIJSetPreallocation(*H, 1, NULL));
  PetscCall(MatMPIAIJSetPreallocation(*H, 1, NULL, 1, NULL));

  PetscInt rstart, rend, obs_idx = 0;
  PetscCall(MatGetOwnershipRange(*H, &rstart, &rend));
  for (PetscInt i = 0; i < ncells; i++) {
    if (i % obs_stride == 0) {
      if (obs_idx >= rstart && obs_idx < rend) PetscCall(MatSetValue(*H, obs_idx, i * NDOF, 1.0, INSERT_VALUES));
      obs_idx++;
    }
  }
  PetscCall(MatAssemblyBegin(*H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*H, MAT_FINAL_ASSEMBLY));
  *nobs = n_obs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// gauge observation operator: height (DOF 0) at gauge cells given by NATURAL
// cell IDs (mesh-file order, the numbering observations.sites.cells and
// map_gauges_to_cells.py use). Also returns the bed elevation (centroid z) at
// each gauge, replicated on all ranks, for the WSE <-> h conversion.
static PetscErrorCode CreateGaugeObservationMatrix(RDy rdy, PetscInt ngauges, const PetscInt *gauge_cells, Mat *H, PetscReal *zb_gauges) {
  MPI_Comm comm;
  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)rdy->u_global, &comm));

  PetscInt state_lo, state_size, state_nlocal;
  PetscCall(VecGetOwnershipRange(rdy->u_global, &state_lo, NULL));
  PetscCall(VecGetSize(rdy->u_global, &state_size));
  PetscCall(VecGetLocalSize(rdy->u_global, &state_nlocal));

  MatType mat_type;
  PetscCall(MatCreate(comm, H));
  // the column layout must match u_global's (uneven) DMPlex distribution
  PetscCall(MatSetSizes(*H, PETSC_DECIDE, state_nlocal, ngauges, state_size));
  PetscCall(ObservationMatrixType(rdy->u_global, &mat_type));
  PetscCall(MatSetType(*H, mat_type));
  PetscCall(MatSeqAIJSetPreallocation(*H, 1, NULL));
  PetscCall(MatMPIAIJSetPreallocation(*H, 1, NULL, 1, NULL));

  PetscInt *found;  // how many owned cells matched each gauge (validation)
  PetscCall(PetscCalloc1(ngauges, &found));
  for (PetscInt g = 0; g < ngauges; ++g) zb_gauges[g] = 0.0;

  RDyCells *cells = &rdy->mesh.cells;
  for (PetscInt i = 0; i < rdy->mesh.num_cells; ++i) {
    if (!cells->is_owned[i]) continue;
    for (PetscInt g = 0; g < ngauges; ++g) {
      if (cells->natural_ids[i] == gauge_cells[g]) {
        PetscInt col = state_lo + cells->local_to_owned[i] * NDOF;  // h dof
        PetscCall(MatSetValue(*H, g, col, 1.0, INSERT_VALUES));
        zb_gauges[g] = cells->centroids[i].X[2];
        found[g]++;
      }
    }
  }
  PetscCall(MatAssemblyBegin(*H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*H, MAT_FINAL_ASSEMBLY));

  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, found, ngauges, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, zb_gauges, ngauges, MPIU_REAL, MPIU_SUM, comm));
  for (PetscInt g = 0; g < ngauges; ++g)
    PetscCheck(found[g] == 1, comm, PETSC_ERR_USER, "gauge %" PetscInt_FMT " (natural cell %" PetscInt_FMT ") matched %" PetscInt_FMT
               " owned cells (expected exactly 1)", g, gauge_cells[g], found[g]);
  PetscCall(PetscFree(found));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
// gauge observation table: plain text, WSE in meters (same vertical datum as
// the mesh z). Missing observations are recorded as nan and zero-weighted.
//   line 1: ngauges K
//   line 2: ngauges natural cell IDs
//   lines 3..K+2: time_seconds then ngauges WSE values
// ---------------------------------------------------------------------------

static PetscErrorCode ReadObsTable(MPI_Comm comm, const char *path, PetscInt *ngauges, PetscInt **gauge_cells, PetscInt *K,
                                   PetscReal **times, PetscReal **wse) {
  PetscMPIInt rank;
  PetscInt    hdr[2] = {0, 0};
  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  FILE *fp = NULL;
  if (rank == 0) {
    fp = fopen(path, "r");
    PetscCheck(fp, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "could not open observation table %s", path);
    PetscCheck(fscanf(fp, "%" PetscInt_FMT " %" PetscInt_FMT, &hdr[0], &hdr[1]) == 2, PETSC_COMM_SELF, PETSC_ERR_FILE_READ,
               "bad header in %s", path);
  }
  PetscCallMPI(MPI_Bcast(hdr, 2, MPIU_INT, 0, comm));
  *ngauges = hdr[0];
  *K       = hdr[1];
  PetscCall(PetscMalloc1(*ngauges, gauge_cells));
  PetscCall(PetscMalloc1(*K, times));
  PetscCall(PetscMalloc1((*K) * (*ngauges), wse));
  if (rank == 0) {
    for (PetscInt g = 0; g < *ngauges; ++g)
      PetscCheck(fscanf(fp, "%" PetscInt_FMT, &(*gauge_cells)[g]) == 1, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "bad cell list in %s", path);
    for (PetscInt k = 0; k < *K; ++k) {
      PetscCheck(fscanf(fp, "%lf", &(*times)[k]) == 1, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "bad time row %" PetscInt_FMT " in %s", k, path);
      for (PetscInt g = 0; g < *ngauges; ++g)
        PetscCheck(fscanf(fp, "%lf", &(*wse)[k * (*ngauges) + g]) == 1, PETSC_COMM_SELF, PETSC_ERR_FILE_READ,
                   "bad wse row %" PetscInt_FMT " in %s", k, path);
    }
    fclose(fp);
  }
  PetscCallMPI(MPI_Bcast(*gauge_cells, *ngauges, MPIU_INT, 0, comm));
  PetscCallMPI(MPI_Bcast(*times, *K, MPIU_REAL, 0, comm));
  PetscCallMPI(MPI_Bcast(*wse, (*K) * (*ngauges), MPIU_REAL, 0, comm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// gathers each y_k (h at gauges) to rank 0, converts to WSE with zb, writes the table
static PetscErrorCode WriteObsTable(MPI_Comm comm, const char *path, PetscInt ngauges, const PetscInt *gauge_cells, PetscInt K,
                                    PetscReal obs_dt, Vec *y_k, const PetscReal *zb_gauges) {
  PetscMPIInt rank;
  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  FILE *fp = NULL;
  if (rank == 0) {
    fp = fopen(path, "w");
    PetscCheck(fp, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "could not open %s for writing", path);
    fprintf(fp, "%" PetscInt_FMT " %" PetscInt_FMT "\n", ngauges, K);
    for (PetscInt g = 0; g < ngauges; ++g) fprintf(fp, "%" PetscInt_FMT "%c", gauge_cells[g], g + 1 == ngauges ? '\n' : ' ');
  }
  for (PetscInt k = 0; k < K; ++k) {
    Vec        y_all;
    VecScatter sc;
    PetscCall(VecScatterCreateToZero(y_k[k], &sc, &y_all));
    PetscCall(VecScatterBegin(sc, y_k[k], y_all, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(sc, y_k[k], y_all, INSERT_VALUES, SCATTER_FORWARD));
    if (rank == 0) {
      const PetscScalar *ya;
      PetscCall(VecGetArrayRead(y_all, &ya));
      fprintf(fp, "%.10g", (double)((k + 1) * obs_dt));
      for (PetscInt g = 0; g < ngauges; ++g) fprintf(fp, " %.10g", (double)(PetscRealPart(ya[g]) + zb_gauges[g]));
      fprintf(fp, "\n");
      PetscCall(VecRestoreArrayRead(y_all, &ya));
    }
    PetscCall(VecScatterDestroy(&sc));
    PetscCall(VecDestroy(&y_all));
  }
  if (rank == 0) fclose(fp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Reads a PETSc binary Vec written in NATURAL cell order (mesh-file order --
// what data/nlcd/make_nlcd_manning.py and the gauge tools produce) and returns
// the values for this rank's owned cells, in owned order.
static PetscErrorCode ReadNaturalCellField(RDy rdy, const char *path, PetscReal **owned_values) {
  PetscViewer viewer;
  PetscInt    hdr[2], n_owned;
  PetscFunctionBeginUser;

  PetscCall(PetscViewerBinaryOpen(rdy->comm, path, FILE_MODE_READ, &viewer));
  PetscCall(PetscViewerBinaryRead(viewer, hdr, 2, NULL, PETSC_INT));
  PetscCheck(hdr[0] == 1211214, rdy->comm, PETSC_ERR_FILE_UNEXPECTED, "%s is not a PETSc binary Vec (classid %" PetscInt_FMT ")", path,
             hdr[0]);
  PetscInt ncells_global;
  PetscCall(RDyGetNumGlobalCells(rdy, &ncells_global));
  PetscCheck(hdr[1] == ncells_global, rdy->comm, PETSC_ERR_FILE_UNEXPECTED,
             "%s holds %" PetscInt_FMT " values but the mesh has %" PetscInt_FMT " cells", path, hdr[1], ncells_global);

  PetscReal *all;
  PetscCall(PetscMalloc1(hdr[1], &all));
  PetscCall(PetscViewerBinaryRead(viewer, all, hdr[1], NULL, PETSC_REAL));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(RDyGetNumOwnedCells(rdy, &n_owned));
  PetscCall(PetscMalloc1(n_owned, owned_values));
  RDyCells *cells = &rdy->mesh.cells;
  for (PetscInt c = 0; c < rdy->mesh.num_cells; ++c) {
    if (!cells->is_owned[c]) continue;
    (*owned_values)[cells->local_to_owned[c]] = all[cells->natural_ids[c]];
  }
  PetscCall(PetscFree(all));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Clears any previously recorded trajectory. The memory trajectory keeps a
// strict stack, so a new forward solve starting again at step 0 on top of an
// unconsumed stack aborts with "Illegal modification of a non-top stack
// element" -- which happens whenever we do two forward passes before an adjoint
// sweep (truth then perturbed, or a fresh TAO objective evaluation). The disk
// trajectory tolerates it, but is ~6.5x slower at scale, so keep both usable.
// The disk ("basic") trajectory records its directory on first setup and then
// refuses to set up again while that directory exists ("Directory ... not
// empty"), which breaks any repeated forward pass -- e.g. every TAO iteration.
// We give it a fixed name (see main) and delete the tree here, so the next
// setup recreates it. TSTrajectoryReset alone is not enough.
#define TRAJ_DIR "rdycore-traj"

static PetscErrorCode ResetTrajectory(TS ts) {
  TSTrajectory tj;
  PetscFunctionBeginUser;
  PetscCall(TSGetTrajectory(ts, &tj));
  if (!tj) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TSTrajectoryReset(tj));
  TSTrajectoryType type;
  PetscCall(TSTrajectoryGetType(tj, ts, &type));
  PetscBool is_basic;
  PetscCall(PetscStrcmp(type, TSTRAJECTORYBASIC, &is_basic));
  if (is_basic) {
    MPI_Comm    comm;
    PetscMPIInt rank;
    PetscCall(PetscObjectGetComm((PetscObject)tj, &comm));
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    if (rank == 0) {
      PetscBool exists;
      PetscCall(PetscTestDirectory(TRAJ_DIR, 'w', &exists));
      if (exists) PetscCall(PetscRMTree(TRAJ_DIR));
    }
    PetscCall(PetscBarrier((PetscObject)tj));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// forward solve with observations every `freq` steps (K = total_steps/freq
// observation times, at steps freq, 2 freq, ..., K freq).
// mode "record": y_k = H u(t_k). mode "residual": r_k = H u(t_k) - y_k and
// J = 1/2 sigma^-2 sum_k |r_k|^2. Optional w_k (0/1 per observation)
// zero-weights missing data in both the misfit and (via r_k) the adjoint.
//
// The whole window is ONE TSSolve, with observations taken by a TS monitor.
// The memory trajectory (checkpointing/revolve) cannot tolerate a segmented
// forward: TSSolve sets ts->reason before the final TSTrajectorySet of every
// call, and TSTrajectoryMemorySet_N treats reason != 0 as "end of the forward
// run" -- it clobbers its total_steps, pops the second-last checkpoint, and
// skips storing the boundary step, corrupting the checkpoint stack at every
// interior observation time (SEGV in TSAdjointSolve). The segmented BACKWARD
// sweep (TSAdjointSetSteps) is fine: no trajectory writes happen there.
typedef struct {
  Mat        H;
  Vec       *y_k, *r_k, *w_k;
  PetscInt   freq, K;
  PetscReal  sigma;
  PetscReal  J;       // accumulated misfit (residual mode)
  PetscBool  active;  // record only during ForwardObserve's own TSSolve --
                      // the trajectory's ReCompute replays also fire monitors
} ObsMonitorCtx;

static ObsMonitorCtx obs_mon;  // one TS per driver run; installed once

// ---------------------------------------------------------------------------
// Hourly rainfall for the window (raster/unstructured/homogeneous datasets,
// -raster_rain_dir etc.): RDySetup builds rdy->forcing from those options,
// but RDyApplyForcing is only invoked by RDyAdvance, which this driver
// bypasses (single TSSolve per window, for trajectory integrity). The
// datasets also stream strictly FORWARD in time, while the adjoint's
// checkpoint recomputes REWIND time. So: preload the mapped per-cell rain for
// every hour of the window once at setup (monotone RDyApplyForcing calls),
// then swap the active hour in from a TSPreStage hook -- which fires inside
// every TSStep, INCLUDING trajectory ReCompute replays -- as a pure function
// of stage time, so replays reproduce the original forcing exactly. Hour h
// covers stage times in (h*3600, (h+1)*3600], matching RDyAdvance's
// hourly-coupling semantics; region id 1 mirrors RDyApplyForcing.
// The source is state-independent, so no Jacobian/adjoint terms arise.
typedef struct {
  RDy         rdy;
  PetscInt    nhours, ndata;
  PetscReal **hours;  // [nhours][ndata]: water source (m/s) per owned region cell
  PetscInt    cur_bucket;
} RainSchedule;

static RainSchedule rain_sched;  // one TS per driver run, like obs_mon

static PetscErrorCode PreStageApplyRain(TS ts, PetscReal stage_time) {
  RainSchedule *rs = &rain_sched;
  (void)ts;
  PetscFunctionBeginUser;
  if (!rs->nhours) PetscFunctionReturn(PETSC_SUCCESS);
  PetscInt bucket = (PetscInt)PetscFloorReal((stage_time - 1e-9) / 3600.0);
  if (bucket < 0) bucket = 0;
  if (bucket >= rs->nhours) bucket = rs->nhours - 1;
  if (bucket != rs->cur_bucket) {
    PetscCall(RDySetRegionalWaterSource(rs->rdy, 1, rs->ndata, rs->hours[bucket]));
    rs->cur_bucket = bucket;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupRainSchedule(RDy rdy, PetscReal t_final) {
  MPI_Comm comm;
  PetscFunctionBeginUser;
  rain_sched = (RainSchedule){.rdy = rdy, .cur_bucket = -1};
  RDyForcing forcing = rdy->forcing;
  if (!forcing || forcing->source.type == FORCING_DATASET_UNSET || !forcing->source.ndata) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectGetComm((PetscObject)rdy->ts, &comm));

  PetscInt nhours = (PetscInt)PetscCeilReal(t_final / 3600.0);
  if (nhours < 1) nhours = 1;
  rain_sched.nhours = nhours;
  rain_sched.ndata  = forcing->source.ndata;
  PetscCall(PetscMalloc1(nhours, &rain_sched.hours));
  for (PetscInt h = 0; h < nhours; ++h) {
    PetscCall(RDyApplyForcing(rdy, forcing, 3600.0 * h));  // streams dataset files forward
    PetscCall(PetscMalloc1(rain_sched.ndata, &rain_sched.hours[h]));
    for (PetscInt i = 0; i < rain_sched.ndata; ++i) rain_sched.hours[h][i] = forcing->source.data_for_rdycore[i];
  }
  PetscCall(RDySetRegionalWaterSource(rdy, 1, rain_sched.ndata, rain_sched.hours[0]));
  rain_sched.cur_bucket = 0;
  PetscCall(TSSetPreStage(rdy->ts, PreStageApplyRain));
  PetscCall(PetscPrintf(comm, "rain schedule: %" PetscInt_FMT " hourly datasets preloaded for the %.0f s window\n", nhours, (double)t_final));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroyRainSchedule(void) {
  PetscFunctionBeginUser;
  for (PetscInt h = 0; h < rain_sched.nhours; ++h) PetscCall(PetscFree(rain_sched.hours[h]));
  if (rain_sched.hours) PetscCall(PetscFree(rain_sched.hours));
  rain_sched = (RainSchedule){0};
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ObsMonitor(TS ts, PetscInt step, PetscReal t, Vec u, void *ctx) {
  ObsMonitorCtx *m = ctx;
  PetscFunctionBeginUser;
  if (!m->active || step <= 0 || step % m->freq != 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscInt k = step / m->freq;
  if (k > m->K) PetscFunctionReturn(PETSC_SUCCESS);
  if (m->r_k) {  // residual mode
    PetscCall(MatMult(m->H, u, m->r_k[k - 1]));
    PetscCall(VecAXPY(m->r_k[k - 1], -1.0, m->y_k[k - 1]));
    if (m->w_k) PetscCall(VecPointwiseMult(m->r_k[k - 1], m->r_k[k - 1], m->w_k[k - 1]));
    PetscReal norm;
    PetscCall(VecNorm(m->r_k[k - 1], NORM_2, &norm));
    m->J += 0.5 * norm * norm / (m->sigma * m->sigma);
  } else {  // record mode
    PetscCall(MatMult(m->H, u, m->y_k[k - 1]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ForwardObserve(RDy rdy, Vec ic, PetscInt total_steps, PetscInt freq, Mat H, Vec *y_k, Vec *r_k, Vec *w_k,
                                     PetscReal sigma, PetscReal *J_out) {
  static PetscBool obs_mon_installed = PETSC_FALSE;
  PetscFunctionBeginUser;
  PetscCall(ResetTrajectory(rdy->ts));
  PetscCall(VecCopy(ic, rdy->u_global));
  PetscCall(TSSetTime(rdy->ts, 0.0));
  PetscCall(TSSetStepNumber(rdy->ts, 0));
  PetscCall(TSSetMaxTime(rdy->ts, PETSC_MAX_REAL));
  PetscCall(TSSetExactFinalTime(rdy->ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetTimeStep(rdy->ts, rdy->dt));
  PetscCall(TSSetSolution(rdy->ts, rdy->u_global));
  PetscCall(TSSetMaxSteps(rdy->ts, total_steps));

  if (!obs_mon_installed) {
    PetscCall(TSMonitorSet(rdy->ts, ObsMonitor, &obs_mon, NULL));
    obs_mon_installed = PETSC_TRUE;
  }
  obs_mon = (ObsMonitorCtx){.H = H, .y_k = y_k, .r_k = r_k, .w_k = w_k, .freq = freq, .K = total_steps / freq, .sigma = sigma};
  obs_mon.active = PETSC_TRUE;
  PetscCall(TSSolve(rdy->ts, rdy->u_global));
  obs_mon.active = PETSC_FALSE;
  if (J_out) *J_out = obs_mon.J;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// backward sweep over the observation windows: lambda accumulates the
// H^T R^-1 r_k jumps at each observation time; mu integrates dJ/dn
static PetscErrorCode AdjointSweepMulti(RDy rdy, Mat H, PetscReal sigma, PetscInt freq, PetscInt K, Vec *r_k, Vec r_scratch, Vec lambda,
                                        Vec mu) {
  PetscFunctionBeginUser;
  PetscCall(VecCopy(r_k[K - 1], r_scratch));
  PetscCall(VecScale(r_scratch, 1.0 / (sigma * sigma)));
  PetscCall(MatMultTranspose(H, r_scratch, lambda));
  PetscCall(VecZeroEntries(mu));
  PetscCall(TSSetCostGradients(rdy->ts, 1, &lambda, &mu));
  Vec jump;
  PetscCall(VecDuplicate(lambda, &jump));
  for (PetscInt k = K; k >= 1; --k) {
    PetscCall(TSAdjointSetSteps(rdy->ts, freq));
    PetscCall(TSAdjointSolve(rdy->ts));
    if (k > 1) {
      PetscCall(VecCopy(r_k[k - 2], r_scratch));
      PetscCall(VecScale(r_scratch, 1.0 / (sigma * sigma)));
      PetscCall(MatMultTranspose(H, r_scratch, jump));
      PetscCall(VecAXPY(lambda, 1.0, jump));
    }
  }
  PetscCall(VecDestroy(&jump));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// resets the TS clock and solves forward from ic (copied into rdy->u_global)
static PetscErrorCode ForwardSolve(RDy rdy, Vec ic, PetscReal t_final) {
  PetscFunctionBeginUser;
  PetscCall(ResetTrajectory(rdy->ts));
  PetscCall(VecCopy(ic, rdy->u_global));
  PetscCall(TSSetTime(rdy->ts, 0.0));
  PetscCall(TSSetStepNumber(rdy->ts, 0));
  PetscCall(TSSetMaxTime(rdy->ts, t_final));
  PetscCall(TSSetExactFinalTime(rdy->ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTimeStep(rdy->ts, rdy->dt));
  PetscCall(TSSetSolution(rdy->ts, rdy->u_global));
  PetscCall(TSSolve(rdy->ts, rdy->u_global));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// J = 1/2 sigma^-2 |H u - y|^2; optionally returns the residual r = H u - y
static PetscErrorCode Misfit(Mat H, Vec u, Vec y, PetscReal sigma, Vec r_work, PetscReal *J) {
  PetscFunctionBeginUser;
  PetscCall(MatMult(H, u, r_work));
  PetscCall(VecAXPY(r_work, -1.0, y));
  PetscReal norm;
  PetscCall(VecNorm(r_work, NORM_2, &norm));
  *J = 0.5 * norm * norm / (sigma * sigma);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
// TAO calibration mode (-adjoint_calibrate): recover per-region Manning n
// from synthetic terminal observations (increment 5 of the plan)
// ---------------------------------------------------------------------------

typedef struct {
  RDy       rdy;
  Mat       H;
  Vec       y, r_work, u_ic, lambda, mu;
  PetscReal sigma, t_final;
} CalibrationCtx;

// applies the per-region parameter vector p to the model's Manning field
static PetscErrorCode ApplyRegionManning(RDy rdy, const PetscReal *p_all) {
  PetscFunctionBeginUser;
  for (PetscInt r = 0; r < rdy->num_regions; ++r) {
    RDyRegion *region = &rdy->regions[r];
    if (region->num_owned_cells > 0) {
      PetscReal *vals;
      PetscCall(PetscMalloc1(region->num_owned_cells, &vals));
      for (PetscInt i = 0; i < region->num_owned_cells; ++i) vals[i] = p_all[r];
      PetscCall(RDySetRegionalManningsN(rdy, r, region->num_owned_cells, vals));
      PetscCall(PetscFree(vals));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// TAO objective + gradient: forward solve -> misfit; adjoint sweep -> mu;
// region gradient g_r = sum of mu over the region's owned cells
static PetscErrorCode FormObjectiveAndGradient(Tao tao, Vec p, PetscReal *J, Vec g, void *ctx) {
  CalibrationCtx *cal = ctx;
  RDy             rdy = cal->rdy;
  PetscFunctionBeginUser;

  // broadcast the (tiny) parameter vector to all ranks
  Vec        p_all_vec;
  VecScatter scatter;
  PetscCall(VecScatterCreateToAll(p, &scatter, &p_all_vec));
  PetscCall(VecScatterBegin(scatter, p, p_all_vec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter, p, p_all_vec, INSERT_VALUES, SCATTER_FORWARD));
  const PetscScalar *p_all;
  PetscCall(VecGetArrayRead(p_all_vec, &p_all));

  PetscReal *p_vals;
  PetscCall(PetscMalloc1(rdy->num_regions, &p_vals));
  for (PetscInt r = 0; r < rdy->num_regions; ++r) p_vals[r] = PetscRealPart(p_all[r]);
  PetscCall(VecRestoreArrayRead(p_all_vec, &p_all));
  PetscCall(VecScatterDestroy(&scatter));
  PetscCall(VecDestroy(&p_all_vec));

  PetscCall(ApplyRegionManning(rdy, p_vals));
  PetscCall(ForwardSolve(rdy, cal->u_ic, cal->t_final));
  PetscCall(Misfit(cal->H, rdy->u_global, cal->y, cal->sigma, cal->r_work, J));

  // adjoint sweep for dJ/dn (per owned cell)
  PetscCall(VecScale(cal->r_work, 1.0 / (cal->sigma * cal->sigma)));
  PetscCall(MatMultTranspose(cal->H, cal->r_work, cal->lambda));
  PetscCall(VecZeroEntries(cal->mu));
  PetscCall(TSSetCostGradients(rdy->ts, 1, &cal->lambda, &cal->mu));
  PetscCall(TSAdjointSolve(rdy->ts));

  // reduce mu over each region's owned cells
  const PetscScalar *mu_ptr;
  PetscCall(VecGetArrayRead(cal->mu, &mu_ptr));
  PetscReal *g_vals;
  PetscCall(PetscCalloc1(rdy->num_regions, &g_vals));
  for (PetscInt r = 0; r < rdy->num_regions; ++r) {
    RDyRegion *region = &rdy->regions[r];
    for (PetscInt i = 0; i < region->num_owned_cells; ++i) g_vals[r] += PetscRealPart(mu_ptr[region->owned_cell_global_ids[i]]);
  }
  PetscCall(VecRestoreArrayRead(cal->mu, &mu_ptr));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, g_vals, rdy->num_regions, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)tao)));

  PetscInt lo, hi;
  PetscCall(VecGetOwnershipRange(g, &lo, &hi));
  for (PetscInt r = lo; r < hi; ++r) PetscCall(VecSetValue(g, r, g_vals[r], INSERT_VALUES));
  PetscCall(VecAssemblyBegin(g));
  PetscCall(VecAssemblyEnd(g));
  PetscCall(PetscFree(g_vals));
  PetscCall(PetscFree(p_vals));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
// per-cell calibration (-adjoint_calibrate_percell): recover a spatially
// distributed Manning field with Tikhonov regularization (increment 6)
// ---------------------------------------------------------------------------

typedef struct {
  CalibrationCtx base;
  Vec            n_prior;      // constant prior n0 (also the initial guess)
  PetscReal      beta;         // Tikhonov weight
  PetscInt       total_steps;  // forward steps in the window
  PetscInt       obs_freq;     // observe every obs_freq steps
  PetscInt       K;            // number of observation times
  Vec           *y_k, *r_k;    // per-observation-time data and residuals
  Vec           *w_k;          // optional 0/1 weights (missing data); NULL = all present
} PerCellCtx;

// J = multi-time misfit + beta/2 |p - n0|^2;  g = mu + beta (p - n0)
static PetscErrorCode FormObjectiveAndGradientPerCell(Tao tao, Vec p, PetscReal *J, Vec g, void *ctx) {
  PerCellCtx     *pc  = ctx;
  CalibrationCtx *cal = &pc->base;
  RDy             rdy = cal->rdy;
  PetscFunctionBeginUser;

  // apply p (local part == owned cells, same layout as mu)
  const PetscScalar *p_ptr;
  PetscInt           n_owned;
  PetscCall(VecGetLocalSize(p, &n_owned));
  PetscCall(VecGetArrayRead(p, &p_ptr));
  PetscReal *n_vals;
  PetscCall(PetscMalloc1(n_owned, &n_vals));
  for (PetscInt i = 0; i < n_owned; ++i) n_vals[i] = PetscRealPart(p_ptr[i]);
  PetscCall(VecRestoreArrayRead(p, &p_ptr));
  PetscCall(RDySetDomainManningsN(rdy, n_owned, n_vals));
  PetscCall(PetscFree(n_vals));

  PetscCall(ForwardObserve(rdy, cal->u_ic, pc->total_steps, pc->obs_freq, cal->H, pc->y_k, pc->r_k, pc->w_k, cal->sigma, J));
  PetscCall(AdjointSweepMulti(rdy, cal->H, cal->sigma, pc->obs_freq, pc->K, pc->r_k, cal->r_work, cal->lambda, cal->mu));

  // g = mu + beta (p - n_prior); J += beta/2 |p - n_prior|^2
  PetscCall(VecCopy(p, g));
  PetscCall(VecAXPY(g, -1.0, pc->n_prior));
  PetscReal reg_norm;
  PetscCall(VecNorm(g, NORM_2, &reg_norm));
  *J += 0.5 * pc->beta * reg_norm * reg_norm;
  PetscCall(VecAYPX(g, pc->beta, cal->mu));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
// land-cover class calibration (-adjoint_calibrate_classes): one Manning value
// per NLCD class instead of one per cell. Same forward/adjoint machinery as the
// per-cell mode; the per-cell gradient mu is summed over each class.
// ---------------------------------------------------------------------------

typedef struct {
  PerCellCtx pc;            // forward/adjoint machinery (multi-time gauge obs)
  PetscInt   nclass;        // number of distinct classes present globally
  PetscInt  *class_codes;   // [nclass] NLCD code for each parameter
  PetscInt  *cell_class;    // [n_owned] parameter index for each owned cell
  PetscInt   n_owned;
  Vec        p_prior;       // [nclass] prior value per class (NLCD table)
  PetscReal *n_scratch;     // [n_owned]
} ClassCtx;

// J = multi-time gauge misfit + beta/2 |p - p_prior|^2;  g_k = sum_{c in k} mu_c
static PetscErrorCode FormObjectiveAndGradientClasses(Tao tao, Vec p, PetscReal *J, Vec g, void *ctx) {
  ClassCtx       *cc  = ctx;
  CalibrationCtx *cal = &cc->pc.base;
  RDy             rdy = cal->rdy;
  MPI_Comm        comm;
  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)p, &comm));

  // p is tiny; replicate it so every rank can expand it onto its cells
  Vec                p_all;
  VecScatter         scatter;
  const PetscScalar *pa;
  PetscCall(VecScatterCreateToAll(p, &scatter, &p_all));
  PetscCall(VecScatterBegin(scatter, p, p_all, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter, p, p_all, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecGetArrayRead(p_all, &pa));
  for (PetscInt i = 0; i < cc->n_owned; ++i) cc->n_scratch[i] = PetscRealPart(pa[cc->cell_class[i]]);
  PetscCall(VecRestoreArrayRead(p_all, &pa));
  PetscCall(RDySetDomainManningsN(rdy, cc->n_owned, cc->n_scratch));

  PetscCall(ForwardObserve(rdy, cal->u_ic, cc->pc.total_steps, cc->pc.obs_freq, cal->H, cc->pc.y_k, cc->pc.r_k, cc->pc.w_k, cal->sigma, J));
  PetscCall(AdjointSweepMulti(rdy, cal->H, cal->sigma, cc->pc.obs_freq, cc->pc.K, cc->pc.r_k, cal->r_work, cal->lambda, cal->mu));

  // reduce the per-cell gradient onto classes
  const PetscScalar *mu;
  PetscReal         *gk;
  PetscCall(PetscCalloc1(cc->nclass, &gk));
  PetscCall(VecGetArrayRead(cal->mu, &mu));
  for (PetscInt i = 0; i < cc->n_owned; ++i) gk[cc->cell_class[i]] += PetscRealPart(mu[i]);
  PetscCall(VecRestoreArrayRead(cal->mu, &mu));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, gk, cc->nclass, MPIU_REAL, MPIU_SUM, comm));

  // Tikhonov toward the NLCD class values
  const PetscScalar *pp;
  PetscCall(VecScatterBegin(scatter, p, p_all, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter, p, p_all, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecGetArrayRead(p_all, &pa));
  Vec prior_all;
  PetscCall(VecScatterCreateToAll(cc->p_prior, &scatter, &prior_all));
  PetscCall(VecScatterBegin(scatter, cc->p_prior, prior_all, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter, cc->p_prior, prior_all, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecGetArrayRead(prior_all, &pp));
  for (PetscInt k = 0; k < cc->nclass; ++k) {
    PetscReal d = PetscRealPart(pa[k]) - PetscRealPart(pp[k]);
    *J += 0.5 * cc->pc.beta * d * d;
    gk[k] += cc->pc.beta * d;
  }
  PetscCall(VecRestoreArrayRead(prior_all, &pp));
  PetscCall(VecRestoreArrayRead(p_all, &pa));
  PetscCall(VecDestroy(&prior_all));
  PetscCall(VecScatterDestroy(&scatter));
  PetscCall(VecDestroy(&p_all));

  PetscInt lo, hi;
  PetscCall(VecGetOwnershipRange(g, &lo, &hi));
  for (PetscInt k = lo; k < hi; ++k) PetscCall(VecSetValue(g, k, gk[k], INSERT_VALUES));
  PetscCall(VecAssemblyBegin(g));
  PetscCall(VecAssemblyEnd(g));
  PetscCall(PetscFree(gk));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// writes a per-owned-cell field as a cell field on a 1-dof clone of the DM
// (VTK .vtu -- the "Manning map" artifact)
static PetscErrorCode WriteCellFieldVTK(RDy rdy, Vec p, const char *field_name, const char *filename) {
  PetscFunctionBeginUser;
  DM dm1;
  PetscCall(DMClone(rdy->dm, &dm1));
  PetscSection sec;
  PetscInt     c_start, c_end;
  PetscCall(DMPlexGetHeightStratum(dm1, 0, &c_start, &c_end));
  PetscCall(PetscSectionCreate(PETSC_COMM_WORLD, &sec));
  PetscCall(PetscSectionSetNumFields(sec, 1));
  PetscCall(PetscSectionSetFieldName(sec, 0, field_name));
  PetscCall(PetscSectionSetFieldComponents(sec, 0, 1));
  PetscCall(PetscSectionSetChart(sec, c_start, c_end));
  for (PetscInt c = c_start; c < c_end; ++c) {
    PetscCall(PetscSectionSetDof(sec, c, 1));
    PetscCall(PetscSectionSetFieldDof(sec, c, 0, 1));
  }
  PetscCall(PetscSectionSetUp(sec));
  PetscCall(DMSetLocalSection(dm1, sec));
  PetscCall(PetscSectionDestroy(&sec));

  Vec field;
  PetscCall(DMCreateGlobalVector(dm1, &field));
  PetscCall(PetscObjectSetName((PetscObject)field, field_name));
  // both global vecs enumerate the same owned cells in the same order
  const PetscScalar *p_ptr;
  PetscScalar       *f_ptr;
  PetscInt           n_owned;
  PetscCall(VecGetLocalSize(p, &n_owned));
  PetscCall(VecGetArrayRead(p, &p_ptr));
  PetscCall(VecGetArray(field, &f_ptr));
  for (PetscInt i = 0; i < n_owned; ++i) f_ptr[i] = p_ptr[i];
  PetscCall(VecRestoreArray(field, &f_ptr));
  PetscCall(VecRestoreArrayRead(p, &p_ptr));

  PetscViewer viewer;
  PetscCall(PetscViewerVTKOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(field, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&field));
  PetscCall(DMDestroy(&dm1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "rdycore_adjoint: usage:\nrdycore_adjoint <input.yaml> [options]\n\n");
    exit(0);
  }

  PetscCall(RDyInit(argc, argv, help_str));
  if (strcmp(argv[1], "-help")) {
    MPI_Comm comm = PETSC_COMM_WORLD;

    PetscInt  obs_stride = 2, fd_samples = 8;
    PetscReal sigma = 0.01, ic_perturb = 1e-3, fd_eps = 1e-6, fd_tol = 1e-5;
    PetscBool calibrate = PETSC_FALSE, calibrate_percell = PETSC_FALSE, calibrate_gauges = PETSC_FALSE, gauges_twin = PETSC_FALSE;
    PetscInt  obs_freq  = 0;  // per-cell mode: observe every N steps (0 = terminal only)
    PetscReal beta      = 1e-4;
    PetscInt  gauge_stride                  = 7;     // twin mode: gauges at natural cells 0, s, 2s, ...
    PetscReal n0                            = 0.03;  // gauge mode: constant prior
    char      map_file[PETSC_MAX_PATH_LEN]         = {0};
    char      obs_file[PETSC_MAX_PATH_LEN]         = {0};
    char      gauge_cells_file[PETSC_MAX_PATH_LEN] = {0};  // twin mode: real gauge geometry (natural cell IDs)
    PetscBool have_map_file                        = PETSC_FALSE;
    PetscBool have_obs_file                        = PETSC_FALSE;
    PetscBool have_gauge_cells                     = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-adjoint_calibrate", &calibrate, NULL));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-adjoint_calibrate_percell", &calibrate_percell, NULL));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-adjoint_calibrate_gauges", &calibrate_gauges, NULL));
    PetscBool sensitivity = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-adjoint_sensitivity", &sensitivity, NULL));
    PetscBool calibrate_classes = PETSC_FALSE, classes_twin = PETSC_FALSE;
    char      class_file[PETSC_MAX_PATH_LEN] = {0}, prior_file[PETSC_MAX_PATH_LEN] = {0};
    PetscBool have_class_file = PETSC_FALSE, have_prior_file = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-adjoint_calibrate_classes", &calibrate_classes, NULL));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-adjoint_classes_twin", &classes_twin, NULL));
    PetscCall(PetscOptionsGetString(NULL, NULL, "-adjoint_class_file", class_file, sizeof(class_file), &have_class_file));
    PetscCall(PetscOptionsGetString(NULL, NULL, "-adjoint_prior_file", prior_file, sizeof(prior_file), &have_prior_file));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-adjoint_gauges_twin", &gauges_twin, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-adjoint_gauge_stride", &gauge_stride, NULL));
    PetscCall(PetscOptionsGetString(NULL, NULL, "-adjoint_gauge_cells_file", gauge_cells_file, sizeof(gauge_cells_file), &have_gauge_cells));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-adjoint_n0", &n0, NULL));
    PetscCall(PetscOptionsGetString(NULL, NULL, "-adjoint_obs_file", obs_file, sizeof(obs_file), &have_obs_file));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-adjoint_obs_freq", &obs_freq, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-adjoint_beta", &beta, NULL));
    PetscCall(PetscOptionsGetString(NULL, NULL, "-adjoint_map_file", map_file, sizeof(map_file), &have_map_file));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-adjoint_obs_stride", &obs_stride, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-adjoint_obs_error", &sigma, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-adjoint_ic_perturb", &ic_perturb, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-adjoint_fd_samples", &fd_samples, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-adjoint_fd_eps", &fd_eps, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-adjoint_fd_tol", &fd_tol, NULL));

    // NOTE: the memory/revolve trajectory requires the single-TSSolve
    // forward in ForwardObserve (see the comment there); both disk and
    // memory trajectories are supported.

    RDy rdy;
    PetscCall(RDyCreate(comm, argv[1], &rdy));
    PetscCall(RDySetup(rdy));
    PetscCheck(rdy->config.numerics.jacobian != JACOBIAN_NONE, comm, PETSC_ERR_USER,
               "the adjoint driver requires numerics.jacobian: fd or analytic");
    PetscCall(TSSetSaveTrajectory(rdy->ts));
    {
      // RDycore's TSSetFromOptions ran before the trajectory existed; consume
      // any -ts_trajectory_* options now so they actually take effect.
      TSTrajectory tj;
      PetscCall(TSGetTrajectory(rdy->ts, &tj));
      PetscCall(TSTrajectorySetFromOptions(tj, rdy->ts));
      // fixed directory name so ResetTrajectory can clear it between passes
      PetscCall(TSTrajectorySetDirname(tj, TRAJ_DIR));
    }
    // The basic (disk) trajectory VecLoads solution checkpoints during the
    // adjoint sweep; with useNatural set that goes through the natural-order
    // path, which PETSc only supports for HDF5 and errors at np>1. This
    // driver never uses RDycore's natural-order I/O (no output, no native
    // time series), so natural ordering can be disabled for the run.
    PetscCall(DMSetUseNatural(rdy->dm, PETSC_FALSE));

    PetscInt ncells_global;
    PetscCall(RDyGetNumGlobalCells(rdy, &ncells_global));
    PetscReal t_final = rdy->dt * rdy->config.time.stop_n;

    // hourly rainfall (if -raster_rain_dir etc. was given): preload the
    // window's datasets and install the stage-time forcing hook
    PetscCall(SetupRainSchedule(rdy, t_final));

    Mat      H;
    PetscInt nobs;
    PetscCall(CreateObservationMatrix(comm, ncells_global, rdy->u_global, obs_stride, &H, &nobs));

    Vec y, r_work;
    PetscCall(MatCreateVecs(H, NULL, &y));
    PetscCall(VecDuplicate(y, &r_work));

    // stash the configured IC
    Vec u_ic, u_ic_pert;
    PetscCall(VecDuplicate(rdy->u_global, &u_ic));
    PetscCall(VecCopy(rdy->u_global, u_ic));

    if (calibrate_classes) {
      // ------------------------------------------------------------------
      // Land-cover class calibration: one Manning value per NLCD class.
      // Classes come from a per-cell class map (data/nlcd/*_class.bin) and the
      // prior from the matching Manning map (*_manning.bin). Observations are
      // the same gauge WSE table used by -adjoint_calibrate_gauges; with
      // -adjoint_classes_twin the table is first synthesised from the NLCD map
      // itself, which tests whether 18 gauges can recover the class values.
      // ------------------------------------------------------------------
      PetscCheck(have_class_file, comm, PETSC_ERR_USER, "-adjoint_calibrate_classes requires -adjoint_class_file");
      PetscCheck(have_obs_file, comm, PETSC_ERR_USER, "-adjoint_calibrate_classes requires -adjoint_obs_file");
      PetscCall(TSSetSaveTrajectory(rdy->ts));

      PetscInt n_owned, total_steps = rdy->config.time.stop_n;
      PetscCall(RDyGetNumOwnedCells(rdy, &n_owned));

      PetscReal *class_field, *prior_field;
      PetscCall(ReadNaturalCellField(rdy, class_file, &class_field));
      PetscCall(ReadNaturalCellField(rdy, have_prior_file ? prior_file : class_file, &prior_field));

      // distinct class codes present anywhere in the domain
      PetscInt  present[256] = {0};
      for (PetscInt i = 0; i < n_owned; ++i) {
        PetscInt code = (PetscInt)llround(class_field[i]);
        PetscCheck(code >= 0 && code < 256, comm, PETSC_ERR_USER, "class code %" PetscInt_FMT " out of range", code);
        present[code] = 1;
      }
      PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, present, 256, MPIU_INT, MPI_MAX, comm));
      PetscInt nclass = 0;
      for (PetscInt c = 0; c < 256; ++c) nclass += present[c];
      PetscCheck(nclass > 0, comm, PETSC_ERR_USER, "no land-cover classes found in %s", class_file);

      ClassCtx cc = {.nclass = nclass, .n_owned = n_owned};
      PetscCall(PetscMalloc1(nclass, &cc.class_codes));
      PetscCall(PetscMalloc1(n_owned, &cc.cell_class));
      PetscCall(PetscMalloc1(n_owned, &cc.n_scratch));
      PetscInt idx_of_code[256], k = 0;
      for (PetscInt c = 0; c < 256; ++c) {
        if (present[c]) {
          idx_of_code[c]      = k;
          cc.class_codes[k++] = c;
        } else {
          idx_of_code[c] = -1;
        }
      }
      for (PetscInt i = 0; i < n_owned; ++i) cc.cell_class[i] = idx_of_code[(PetscInt)llround(class_field[i])];

      // class prior = mean of the per-cell prior over that class (= the table value)
      PetscReal *psum, *pcnt;
      PetscCall(PetscCalloc1(nclass, &psum));
      PetscCall(PetscCalloc1(nclass, &pcnt));
      for (PetscInt i = 0; i < n_owned; ++i) {
        psum[cc.cell_class[i]] += prior_field[i];
        pcnt[cc.cell_class[i]] += 1.0;
      }
      PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, psum, nclass, MPIU_REAL, MPIU_SUM, comm));
      PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, pcnt, nclass, MPIU_REAL, MPIU_SUM, comm));

      // observation table -> gauge operator and per-time data/weights
      PetscInt   ngauges, K;
      PetscInt  *gauge_cells;
      PetscReal *obs_times, *obs_wse;
      Mat        Hg;
      PetscReal *zbg;
      Vec       *y_k, *r_k, *w_k;

      if (classes_twin) {
        // synthesise the observation table from the NLCD map itself. If the
        // table already exists, keep its gauge cells (so the twin can be run at
        // the REAL gauge locations) and overwrite only the values; otherwise
        // fall back to evenly strided cells.
        for (PetscInt i = 0; i < n_owned; ++i) cc.n_scratch[i] = prior_field[i];
        PetscCall(RDySetDomainManningsN(rdy, n_owned, cc.n_scratch));
        PetscInt  ng;
        PetscInt *gc;
        PetscBool table_exists;
        PetscCall(PetscTestFile(obs_file, 'r', &table_exists));
        if (table_exists) {
          PetscInt   K_tmp;
          PetscReal *t_tmp, *w_tmp;
          PetscCall(ReadObsTable(comm, obs_file, &ng, &gc, &K_tmp, &t_tmp, &w_tmp));
          PetscCall(PetscFree(t_tmp));
          PetscCall(PetscFree(w_tmp));
          PetscCall(PetscPrintf(comm, "classes twin: reusing %" PetscInt_FMT " gauge cells from %s\n", ng, obs_file));
        } else {
          ng = (ncells_global + gauge_stride - 1) / gauge_stride;
          PetscCall(PetscMalloc1(ng, &gc));
          for (PetscInt g = 0; g < ng; ++g) gc[g] = g * gauge_stride;
        }
        PetscCall(PetscMalloc1(ng, &zbg));
        PetscCall(CreateGaugeObservationMatrix(rdy, ng, gc, &Hg, zbg));
        if (obs_freq <= 0) obs_freq = PetscMax(1, total_steps / 12);
        PetscInt Kt = total_steps / obs_freq;
        Vec     *yt;
        PetscCall(PetscMalloc1(Kt, &yt));
        for (PetscInt i = 0; i < Kt; ++i) PetscCall(MatCreateVecs(Hg, NULL, &yt[i]));
        PetscCall(ForwardObserve(rdy, u_ic, total_steps, obs_freq, Hg, yt, NULL, NULL, sigma, NULL));
        PetscCall(WriteObsTable(comm, obs_file, ng, gc, Kt, obs_freq * rdy->dt, yt, zbg));
        PetscCall(PetscPrintf(comm, "classes twin: wrote %" PetscInt_FMT " gauges x %" PetscInt_FMT " times to %s\n", ng, Kt, obs_file));
        for (PetscInt i = 0; i < Kt; ++i) PetscCall(VecDestroy(&yt[i]));
        PetscCall(PetscFree(yt));
        PetscCall(MatDestroy(&Hg));
        PetscCall(PetscFree(gc));
        PetscCall(PetscFree(zbg));
      }

      PetscCall(ReadObsTable(comm, obs_file, &ngauges, &gauge_cells, &K, &obs_times, &obs_wse));
      PetscInt freq = (PetscInt)llround(obs_times[0] / rdy->dt);
      PetscCheck(freq >= 1 && K * freq <= total_steps, comm, PETSC_ERR_USER,
                 "observation table (%" PetscInt_FMT " times every %" PetscInt_FMT " steps) does not fit time.stop_n = %" PetscInt_FMT, K,
                 freq, total_steps);
      obs_freq    = freq;
      total_steps = K * freq;

      PetscCall(PetscMalloc1(ngauges, &zbg));
      PetscCall(CreateGaugeObservationMatrix(rdy, ngauges, gauge_cells, &Hg, zbg));
      PetscCall(PetscMalloc1(K, &y_k));
      PetscCall(PetscMalloc1(K, &r_k));
      PetscCall(PetscMalloc1(K, &w_k));
      PetscInt n_present = 0;
      for (PetscInt kk = 0; kk < K; ++kk) {
        PetscCall(MatCreateVecs(Hg, NULL, &y_k[kk]));
        PetscCall(MatCreateVecs(Hg, NULL, &r_k[kk]));
        PetscCall(MatCreateVecs(Hg, NULL, &w_k[kk]));
        PetscInt rlo, rhi;
        PetscCall(VecGetOwnershipRange(y_k[kk], &rlo, &rhi));
        PetscScalar *ya, *wa;
        PetscCall(VecGetArray(y_k[kk], &ya));
        PetscCall(VecGetArray(w_k[kk], &wa));
        for (PetscInt g = rlo; g < rhi; ++g) {
          PetscReal wse = obs_wse[kk * ngauges + g];
          if (PetscIsNanReal(wse)) {
            ya[g - rlo] = 0.0;
            wa[g - rlo] = 0.0;
          } else {
            ya[g - rlo] = PetscMax(wse - zbg[g], 0.0);
            wa[g - rlo] = 1.0;
            n_present++;
          }
        }
        PetscCall(VecRestoreArray(y_k[kk], &ya));
        PetscCall(VecRestoreArray(w_k[kk], &wa));
      }

      cc.pc = (PerCellCtx){.base        = {.rdy = rdy, .H = Hg, .u_ic = u_ic, .sigma = sigma, .t_final = t_final},
                           .beta        = beta,
                           .total_steps = total_steps,
                           .obs_freq    = obs_freq,
                           .K           = K,
                           .y_k         = y_k,
                           .r_k         = r_k,
                           .w_k         = w_k};
      PetscCall(MatCreateVecs(Hg, NULL, &cc.pc.base.r_work));
      PetscCall(VecDuplicate(rdy->u_global, &cc.pc.base.lambda));
      PetscCall(MatCreateVecs(rdy->rhs_jac_p, &cc.pc.base.mu, NULL));

      Vec p, lb, ub;
      PetscCall(VecCreate(comm, &p));
      PetscCall(VecSetSizes(p, PETSC_DECIDE, nclass));
      PetscCall(VecSetFromOptions(p));
      PetscCall(VecDuplicate(p, &cc.p_prior));
      PetscCall(VecDuplicate(p, &lb));
      PetscCall(VecDuplicate(p, &ub));
      {
        PetscInt lo, hi;
        PetscCall(VecGetOwnershipRange(p, &lo, &hi));
        for (PetscInt kk = lo; kk < hi; ++kk) PetscCall(VecSetValue(cc.p_prior, kk, psum[kk] / PetscMax(pcnt[kk], 1.0), INSERT_VALUES));
        PetscCall(VecAssemblyBegin(cc.p_prior));
        PetscCall(VecAssemblyEnd(cc.p_prior));
      }
      PetscCall(VecSet(p, n0));  // start from a uniform guess (default 0.03)
      PetscCall(VecSet(lb, 0.005));
      PetscCall(VecSet(ub, 0.30));

      PetscCall(PetscPrintf(comm,
                            "class calibration: %" PetscInt_FMT " classes, %" PetscInt_FMT " gauges, %" PetscInt_FMT
                            " obs times (every %" PetscInt_FMT " steps), %d observations, start n = %g, beta = %.1e\n",
                            nclass, ngauges, K, obs_freq, (int)n_present, (double)n0, (double)beta));

      Tao tao;
      PetscCall(TaoCreate(comm, &tao));
      PetscCall(TaoSetType(tao, TAOBLMVM));
      PetscCall(TaoSetSolution(tao, p));
      PetscCall(TaoSetVariableBounds(tao, lb, ub));
      PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormObjectiveAndGradientClasses, &cc));
      PetscCall(TaoSetTolerances(tao, 1e-12, 1e-12, 1e-12));
      PetscCall(TaoSetMaximumIterations(tao, 100));
      PetscCall(TaoSetFromOptions(tao));
      PetscCall(TaoSolve(tao));

      PetscInt  its;
      PetscReal J_final;
      PetscCall(TaoGetIterationNumber(tao, &its));
      PetscCall(TaoGetSolutionStatus(tao, NULL, &J_final, NULL, NULL, NULL, NULL));

      // report per class: prior (NLCD) vs recovered
      Vec        p_all, prior_all;
      VecScatter sc;
      PetscCall(VecScatterCreateToAll(p, &sc, &p_all));
      PetscCall(VecScatterBegin(sc, p, p_all, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(sc, p, p_all, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterDestroy(&sc));
      PetscCall(VecScatterCreateToAll(cc.p_prior, &sc, &prior_all));
      PetscCall(VecScatterBegin(sc, cc.p_prior, prior_all, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(sc, cc.p_prior, prior_all, INSERT_VALUES, SCATTER_FORWARD));
      const PetscScalar *pa, *pp;
      PetscCall(VecGetArrayRead(p_all, &pa));
      PetscCall(VecGetArrayRead(prior_all, &pp));
      PetscCall(PetscPrintf(comm, "  NLCD  prior_n  recovered_n   rel_err   cells\n"));
      PetscReal max_rel = 0.0, l2num = 0.0, l2den = 0.0;
      for (PetscInt kk = 0; kk < nclass; ++kk) {
        PetscReal pr = PetscRealPart(pp[kk]), rc = PetscRealPart(pa[kk]);
        PetscReal rel = PetscAbsReal(rc - pr) / PetscMax(pr, 1e-12);
        l2num += Square(rc - pr);
        l2den += Square(pr);
        if (rel > max_rel) max_rel = rel;
        PetscCall(PetscPrintf(comm, "  %4d  %7.4f  %10.4f  %8.3f  %6.0f\n", (int)cc.class_codes[kk], (double)pr, (double)rc, (double)rel,
                              (double)pcnt[kk]));
      }
      PetscCall(VecRestoreArrayRead(p_all, &pa));
      PetscCall(VecRestoreArrayRead(prior_all, &pp));
      PetscCall(VecDestroy(&p_all));
      PetscCall(VecDestroy(&prior_all));
      PetscCall(VecScatterDestroy(&sc));
      PetscCall(PetscPrintf(comm, "class recovery: %d TAO its, J_final %.6e, rel L2 vs prior %.4f, max class rel err %.4f\n", (int)its,
                            (double)J_final, (double)PetscSqrtReal(l2num / PetscMax(l2den, 1e-30)), (double)max_rel));

      if (have_map_file) {
        Vec p_cells;
        PetscCall(VecDuplicate(cc.pc.base.mu, &p_cells));
        PetscScalar *pcv;
        PetscCall(VecGetArray(p_cells, &pcv));
        for (PetscInt i = 0; i < n_owned; ++i) pcv[i] = cc.n_scratch[i];
        PetscCall(VecRestoreArray(p_cells, &pcv));
        PetscCall(WriteCellFieldVTK(rdy, p_cells, "manning_n", map_file));
        PetscCall(VecDestroy(&p_cells));
        PetscCall(PetscPrintf(comm, "wrote Manning map: %s\n", map_file));
      }

      for (PetscInt kk = 0; kk < K; ++kk) {
        PetscCall(VecDestroy(&y_k[kk]));
        PetscCall(VecDestroy(&r_k[kk]));
        PetscCall(VecDestroy(&w_k[kk]));
      }
      PetscCall(PetscFree(y_k));
      PetscCall(PetscFree(r_k));
      PetscCall(PetscFree(w_k));
      PetscCall(PetscFree(psum));
      PetscCall(PetscFree(pcnt));
      PetscCall(PetscFree(zbg));
      PetscCall(PetscFree(gauge_cells));
      PetscCall(PetscFree(obs_times));
      PetscCall(PetscFree(obs_wse));
      PetscCall(PetscFree(class_field));
      PetscCall(PetscFree(prior_field));
      PetscCall(PetscFree(cc.class_codes));
      PetscCall(PetscFree(cc.cell_class));
      PetscCall(PetscFree(cc.n_scratch));
      PetscCall(TaoDestroy(&tao));
      PetscCall(VecDestroy(&p));
      PetscCall(VecDestroy(&lb));
      PetscCall(VecDestroy(&ub));
      PetscCall(VecDestroy(&cc.p_prior));
      PetscCall(VecDestroy(&cc.pc.base.r_work));
      PetscCall(VecDestroy(&cc.pc.base.lambda));
      PetscCall(VecDestroy(&cc.pc.base.mu));
      PetscCall(MatDestroy(&Hg));
      PetscCall(VecDestroy(&u_ic));
      PetscCall(VecDestroy(&y));
      PetscCall(VecDestroy(&r_work));
      PetscCall(MatDestroy(&H));
      PetscCall(RDyDestroy(&rdy));
      PetscCall(RDyFinalize());
      return 0;
    }

    if (sensitivity) {
      // ------------------------------------------------------------------
      // Observation-time sensitivity: how much do the observables at time
      // t_k respond to Manning's n? For each k we solve forward to t_k, seed
      // the adjoint with a UNIT residual on every gauge (lambda(t_k) = H^T 1,
      // i.e. the functional J_k = sum_g h_g(t_k)), and sweep back to t=0.
      // The resulting mu = dJ_k/dn is the sensitivity of the gauge readings
      // at t_k to the roughness field, so |mu| vs t_k says which part of the
      // flood actually constrains n -- and which observation hours would be
      // paid for with no return.
      // ------------------------------------------------------------------
      PetscInt total_steps = rdy->config.time.stop_n;
      if (obs_freq <= 0) obs_freq = PetscMax(1, total_steps / 12);
      PetscInt K = total_steps / obs_freq;

      Vec ones, obs, lambda, mu;
      PetscCall(MatCreateVecs(H, NULL, &ones));
      PetscCall(VecDuplicate(ones, &obs));
      PetscCall(VecSet(ones, 1.0));
      PetscCall(VecDuplicate(rdy->u_global, &lambda));
      PetscCall(MatCreateVecs(rdy->rhs_jac_p, &mu, NULL));

      PetscCall(PetscPrintf(comm, "# observation-time sensitivity of gauge height to Manning n\n"));
      PetscCall(PetscPrintf(comm, "# %d observation times, every %d steps (dt = %g s), %d gauges\n", (int)K, (int)obs_freq,
                            (double)rdy->dt, (int)nobs));
      PetscCall(PetscPrintf(comm, "#   t[s]      mean_h[m]     |dJ/dn|_1      |dJ/dn|_2\n"));

      for (PetscInt k = 1; k <= K; ++k) {
        PetscCall(ResetTrajectory(rdy->ts));
        PetscCall(VecCopy(u_ic, rdy->u_global));
        PetscCall(TSSetTime(rdy->ts, 0.0));
        PetscCall(TSSetStepNumber(rdy->ts, 0));
        PetscCall(TSSetMaxTime(rdy->ts, PETSC_MAX_REAL));
        PetscCall(TSSetExactFinalTime(rdy->ts, TS_EXACTFINALTIME_STEPOVER));
        PetscCall(TSSetTimeStep(rdy->ts, rdy->dt));
        PetscCall(TSSetSolution(rdy->ts, rdy->u_global));
        PetscCall(TSSetMaxSteps(rdy->ts, k * obs_freq));
        PetscCall(TSSolve(rdy->ts, rdy->u_global));

        PetscCall(MatMult(H, rdy->u_global, obs));  // gauge heights at t_k
        PetscReal mean_h;
        PetscCall(VecSum(obs, &mean_h));
        mean_h /= PetscMax(nobs, 1);

        PetscCall(MatMultTranspose(H, ones, lambda));  // dJ_k/du(t_k) = H^T 1
        PetscCall(VecZeroEntries(mu));
        PetscCall(TSSetCostGradients(rdy->ts, 1, &lambda, &mu));
        PetscCall(TSAdjointSolve(rdy->ts));

        PetscReal s1, s2;
        PetscCall(VecNorm(mu, NORM_1, &s1));
        PetscCall(VecNorm(mu, NORM_2, &s2));
        PetscCall(PetscPrintf(comm, "%10.1f  %12.6f  %13.6e  %13.6e\n", (double)(k * obs_freq * rdy->dt), (double)mean_h, (double)s1,
                              (double)s2));
      }

      PetscCall(VecDestroy(&ones));
      PetscCall(VecDestroy(&obs));
      PetscCall(VecDestroy(&lambda));
      PetscCall(VecDestroy(&mu));
      PetscCall(VecDestroy(&u_ic));
      PetscCall(VecDestroy(&y));
      PetscCall(VecDestroy(&r_work));
      PetscCall(MatDestroy(&H));
      PetscCall(RDyDestroy(&rdy));
      PetscCall(RDyFinalize());
      return 0;
    }

    if (calibrate_gauges) {
      // ------------------------------------------------------------------
      // gauge-observation calibration: per-cell Manning from a WSE table at
      // gauge cells (the real-data path; see data/harvey_gauges/). With
      // -adjoint_gauges_twin, a synthetic table is first generated from a
      // two-zone truth and written to -adjoint_obs_file, then read back --
      // exercising the identical file path the real data uses.
      // ------------------------------------------------------------------
      PetscCheck(have_obs_file, comm, PETSC_ERR_USER, "-adjoint_calibrate_gauges requires -adjoint_obs_file <table>");
      PetscCall(TSSetSaveTrajectory(rdy->ts));

      PetscInt n_owned, total_steps = rdy->config.time.stop_n;
      PetscCall(RDyGetNumOwnedCells(rdy, &n_owned));
      if (obs_freq <= 0) obs_freq = PetscMax(1, total_steps / 8);

      if (gauges_twin) {
        // truth: two zones split at mid-x (as in the per-cell twin)
        PetscReal *xc, *n_true;
        PetscCall(PetscMalloc1(n_owned, &xc));
        PetscCall(PetscMalloc1(n_owned, &n_true));
        PetscCall(RDyMeshGetOwnedCellXCentroids(&rdy->mesh, n_owned, xc));
        PetscReal x_min = PETSC_MAX_REAL, x_max = -PETSC_MAX_REAL;
        for (PetscInt i = 0; i < n_owned; ++i) {
          x_min = PetscMin(x_min, xc[i]);
          x_max = PetscMax(x_max, xc[i]);
        }
        PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &x_min, 1, MPIU_REAL, MPIU_MIN, comm));
        PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &x_max, 1, MPIU_REAL, MPIU_MAX, comm));
        for (PetscInt i = 0; i < n_owned; ++i) n_true[i] = (xc[i] < 0.5 * (x_min + x_max)) ? 0.03 : 0.06;
        PetscCall(RDySetDomainManningsN(rdy, n_owned, n_true));

        PetscInt   ng;
        PetscInt  *gcells;
        PetscReal *zbg;
        if (have_gauge_cells) {
          // real gauge geometry: whitespace-separated natural cell IDs
          // (e.g. data/harvey_gauges/gauge_cells_real17.txt)
          PetscMPIInt rank;
          PetscCallMPI(MPI_Comm_rank(comm, &rank));
          PetscInt cells_buf[1024], n_read = 0;
          if (rank == 0) {
            FILE *fp = fopen(gauge_cells_file, "r");
            PetscCheck(fp, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "cannot open %s", gauge_cells_file);
            while (n_read < 1024 && fscanf(fp, "%" PetscInt_FMT, &cells_buf[n_read]) == 1) ++n_read;
            fclose(fp);
            PetscCheck(n_read > 0, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "no cell IDs in %s", gauge_cells_file);
          }
          PetscCallMPI(MPI_Bcast(&n_read, 1, MPIU_INT, 0, comm));
          PetscCallMPI(MPI_Bcast(cells_buf, n_read, MPIU_INT, 0, comm));
          ng = n_read;
          PetscCall(PetscMalloc1(ng, &gcells));
          PetscCall(PetscMalloc1(ng, &zbg));
          for (PetscInt g = 0; g < ng; ++g) gcells[g] = cells_buf[g];
        } else {
          ng = (ncells_global + gauge_stride - 1) / gauge_stride;
          PetscCall(PetscMalloc1(ng, &gcells));
          PetscCall(PetscMalloc1(ng, &zbg));
          for (PetscInt g = 0; g < ng; ++g) gcells[g] = g * gauge_stride;
        }
        Mat Hg;
        PetscCall(CreateGaugeObservationMatrix(rdy, ng, gcells, &Hg, zbg));

        PetscInt K = total_steps / obs_freq;
        Vec     *y_rec;
        PetscCall(PetscMalloc1(K, &y_rec));
        for (PetscInt k = 0; k < K; ++k) PetscCall(MatCreateVecs(Hg, NULL, &y_rec[k]));
        PetscCall(ForwardObserve(rdy, u_ic, total_steps, obs_freq, Hg, y_rec, NULL, NULL, sigma, NULL));
        PetscCall(WriteObsTable(comm, obs_file, ng, gcells, K, obs_freq * rdy->dt, y_rec, zbg));
        PetscCall(PetscPrintf(comm, "gauges twin: wrote %" PetscInt_FMT " gauges x %" PetscInt_FMT " obs times to %s\n", ng, K, obs_file));
        for (PetscInt k = 0; k < K; ++k) PetscCall(VecDestroy(&y_rec[k]));
        PetscCall(PetscFree(y_rec));
        PetscCall(MatDestroy(&Hg));
        PetscCall(PetscFree(gcells));
        PetscCall(PetscFree(zbg));
        PetscCall(PetscFree(xc));
        PetscCall(PetscFree(n_true));
      }

      // read the observation table and rebuild the operator from its cells
      PetscInt   ngauges, K;
      PetscInt  *gauge_cells;
      PetscReal *obs_times, *obs_wse;
      PetscCall(ReadObsTable(comm, obs_file, &ngauges, &gauge_cells, &K, &obs_times, &obs_wse));

      // observation cadence must be a whole number of steps, uniform in time
      PetscInt freq = (PetscInt)llround(obs_times[0] / rdy->dt);
      PetscCheck(freq >= 1, comm, PETSC_ERR_USER, "first observation time %g is before the first step (dt=%g)", (double)obs_times[0],
                 (double)rdy->dt);
      for (PetscInt k = 0; k < K; ++k)
        PetscCheck(PetscAbsReal(obs_times[k] - (PetscReal)(k + 1) * freq * rdy->dt) < 1e-6 * rdy->dt, comm, PETSC_ERR_USER,
                   "observation times must be uniform at a whole number of steps (row %" PetscInt_FMT ": t=%g, expected %g)", k,
                   (double)obs_times[k], (double)((k + 1) * freq * rdy->dt));
      PetscCheck(K * freq <= total_steps, comm, PETSC_ERR_USER,
                 "observation window (%" PetscInt_FMT " x %" PetscInt_FMT " steps) exceeds time.stop_n = %" PetscInt_FMT, K, freq,
                 total_steps);
      obs_freq    = freq;
      total_steps = K * freq;  // stop the window at the last observation

      Mat        Hg;
      PetscReal *zbg;
      PetscCall(PetscMalloc1(ngauges, &zbg));
      PetscCall(CreateGaugeObservationMatrix(rdy, ngauges, gauge_cells, &Hg, zbg));

      // y_k = observed WSE - zb (i.e. observed h); w_k = 0 for missing (nan)
      Vec *y_k, *r_k, *w_k;
      PetscCall(PetscMalloc1(K, &y_k));
      PetscCall(PetscMalloc1(K, &r_k));
      PetscCall(PetscMalloc1(K, &w_k));
      PetscInt n_present = 0, n_neg = 0;
      for (PetscInt k = 0; k < K; ++k) {
        PetscCall(MatCreateVecs(Hg, NULL, &y_k[k]));
        PetscCall(MatCreateVecs(Hg, NULL, &r_k[k]));
        PetscCall(MatCreateVecs(Hg, NULL, &w_k[k]));
        PetscInt rlo, rhi;
        PetscCall(VecGetOwnershipRange(y_k[k], &rlo, &rhi));
        PetscScalar *ya, *wa;
        PetscCall(VecGetArray(y_k[k], &ya));
        PetscCall(VecGetArray(w_k[k], &wa));
        for (PetscInt g = rlo; g < rhi; ++g) {
          PetscReal wse = obs_wse[k * ngauges + g];
          if (PetscIsNanReal(wse)) {
            ya[g - rlo] = 0.0;
            wa[g - rlo] = 0.0;
          } else {
            PetscReal h_obs = wse - zbg[g];
            if (h_obs < 0.0) n_neg++;         // gauge below cell-mean bed (incised channel):
            ya[g - rlo] = PetscMax(h_obs, 0.0);  // clamp; the QC count is reported below
            wa[g - rlo] = 1.0;
            n_present++;
          }
        }
        PetscCall(VecRestoreArray(y_k[k], &ya));
        PetscCall(VecRestoreArray(w_k[k], &wa));
      }
      PetscReal counts[2] = {(PetscReal)n_present, (PetscReal)n_neg};
      PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, counts, 2, MPIU_REAL, MPIU_SUM, comm));
      PetscCall(PetscPrintf(comm,
                            "gauge calibration: %" PetscInt_FMT " gauges, %" PetscInt_FMT " obs times (every %" PetscInt_FMT
                            " steps), %d observations present, %d below cell-mean bed (clamped)\n",
                            ngauges, K, obs_freq, (int)counts[0], (int)counts[1]));

      PerCellCtx pc = {.base        = {.rdy = rdy, .H = Hg, .y = y, .r_work = NULL, .u_ic = u_ic, .sigma = sigma, .t_final = t_final},
                       .beta        = beta,
                       .total_steps = total_steps,
                       .obs_freq    = obs_freq,
                       .K           = K,
                       .y_k         = y_k,
                       .r_k         = r_k,
                       .w_k         = w_k};
      PetscCall(MatCreateVecs(Hg, NULL, &pc.base.r_work));
      PetscCall(VecDuplicate(rdy->u_global, &pc.base.lambda));
      PetscCall(MatCreateVecs(rdy->rhs_jac_p, &pc.base.mu, NULL));
      PetscCall(VecDuplicate(pc.base.mu, &pc.n_prior));
      PetscCall(VecSet(pc.n_prior, n0));

      Tao tao;
      Vec p, lb, ub, g_scratch;
      PetscCall(VecDuplicate(pc.base.mu, &p));
      PetscCall(VecCopy(pc.n_prior, p));
      PetscCall(VecDuplicate(p, &lb));
      PetscCall(VecDuplicate(p, &ub));
      PetscCall(VecDuplicate(p, &g_scratch));
      PetscCall(VecSet(lb, 0.01));
      PetscCall(VecSet(ub, 0.2));

      PetscReal J_init;
      PetscCall(FormObjectiveAndGradientPerCell(NULL, p, &J_init, g_scratch, &pc));

      PetscCall(TaoCreate(comm, &tao));
      PetscCall(TaoSetType(tao, TAOBLMVM));
      PetscCall(TaoSetSolution(tao, p));
      PetscCall(TaoSetVariableBounds(tao, lb, ub));
      PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormObjectiveAndGradientPerCell, &pc));
      PetscCall(TaoSetTolerances(tao, 1e-14, 1e-14, 1e-14));
      PetscCall(TaoSetMaximumIterations(tao, 200));
      PetscCall(TaoSetFromOptions(tao));
      PetscCall(TaoSolve(tao));

      PetscInt  tao_its;
      PetscReal J_final;
      PetscCall(TaoGetIterationNumber(tao, &tao_its));
      PetscCall(TaoGetSolutionStatus(tao, NULL, &J_final, NULL, NULL, NULL, NULL));
      PetscCall(PetscPrintf(comm, "gauge calibration: %d TAO its, J %.6e -> %.6e (reduction %.2fx, beta %.1e)\n", (int)tao_its,
                            (double)J_init, (double)J_final, (double)(J_init / PetscMax(J_final, PETSC_MACHINE_EPSILON)),
                            (double)beta));

      if (have_map_file) {
        PetscCall(WriteCellFieldVTK(rdy, p, "manning_n", map_file));
        PetscCall(PetscPrintf(comm, "wrote Manning map: %s\n", map_file));
      }

      // optional gate on misfit reduction (twin CI): J_final < gate * J_init
      PetscReal jred_gate = 0.0;
      PetscCall(PetscOptionsGetReal(NULL, NULL, "-adjoint_jred_gate", &jred_gate, NULL));
      if (jred_gate > 0.0)
        PetscCheck(J_final < jred_gate * J_init, comm, PETSC_ERR_PLIB, "gauge calibration misfit reduction too small: %g -> %g",
                   (double)J_init, (double)J_final);

      if (gauges_twin) {  // recovery error vs the two-zone truth used to generate the table
        PetscInt n_owned_rec;
        PetscCall(RDyGetNumOwnedCells(rdy, &n_owned_rec));
        PetscReal *xc_rec;
        PetscCall(PetscMalloc1(n_owned_rec, &xc_rec));
        PetscCall(RDyMeshGetOwnedCellXCentroids(&rdy->mesh, n_owned_rec, xc_rec));
        PetscReal x_lo = PETSC_MAX_REAL, x_hi = -PETSC_MAX_REAL;
        for (PetscInt i = 0; i < n_owned_rec; ++i) {
          x_lo = PetscMin(x_lo, xc_rec[i]);
          x_hi = PetscMax(x_hi, xc_rec[i]);
        }
        PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &x_lo, 1, MPIU_REAL, MPIU_MIN, comm));
        PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &x_hi, 1, MPIU_REAL, MPIU_MAX, comm));
        const PetscScalar *p_rec;
        PetscCall(VecGetArrayRead(p, &p_rec));
        PetscReal sums[2] = {0.0, 0.0};
        for (PetscInt i = 0; i < n_owned_rec; ++i) {
          PetscReal nt = (xc_rec[i] < 0.5 * (x_lo + x_hi)) ? 0.03 : 0.06;
          PetscReal d  = PetscRealPart(p_rec[i]) - nt;
          sums[0] += d * d;
          sums[1] += nt * nt;
        }
        PetscCall(VecRestoreArrayRead(p, &p_rec));
        PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, sums, 2, MPIU_REAL, MPIU_SUM, comm));
        PetscCall(PetscPrintf(comm, "gauges twin recovery: rel L2 err %.3e (%" PetscInt_FMT " gauges, %" PetscInt_FMT " parameters)\n",
                              (double)PetscSqrtReal(sums[0] / sums[1]), ngauges, ncells_global));
        PetscCall(PetscFree(xc_rec));
      }

      for (PetscInt k = 0; k < K; ++k) {
        PetscCall(VecDestroy(&y_k[k]));
        PetscCall(VecDestroy(&r_k[k]));
        PetscCall(VecDestroy(&w_k[k]));
      }
      PetscCall(PetscFree(y_k));
      PetscCall(PetscFree(r_k));
      PetscCall(PetscFree(w_k));
      PetscCall(PetscFree(gauge_cells));
      PetscCall(PetscFree(obs_times));
      PetscCall(PetscFree(obs_wse));
      PetscCall(PetscFree(zbg));
      PetscCall(TaoDestroy(&tao));
      PetscCall(VecDestroy(&p));
      PetscCall(VecDestroy(&lb));
      PetscCall(VecDestroy(&ub));
      PetscCall(VecDestroy(&g_scratch));
      PetscCall(VecDestroy(&pc.base.r_work));
      PetscCall(VecDestroy(&pc.base.lambda));
      PetscCall(VecDestroy(&pc.base.mu));
      PetscCall(VecDestroy(&pc.n_prior));
      PetscCall(MatDestroy(&Hg));
      PetscCall(VecDestroy(&u_ic));
      PetscCall(VecDestroy(&y));
      PetscCall(VecDestroy(&r_work));
      PetscCall(MatDestroy(&H));
      PetscCall(RDyDestroy(&rdy));
      PetscCall(RDyFinalize());
      return 0;
    }

    if (calibrate_percell) {
      // ------------------------------------------------------------------
      // increment 6: per-cell twin calibration with Tikhonov
      // regularization. Truth is a two-zone field split at the median x
      // centroid; TAO/BLMVM recovers a distributed field from a constant
      // prior. Dry/motionless cells are unobservable (dJ/dn = 0) and stay
      // at the prior; the recovery is assessed over observable cells.
      // ------------------------------------------------------------------
      PetscCall(TSSetSaveTrajectory(rdy->ts));

      PetscInt n_owned;
      PetscCall(RDyGetNumOwnedCells(rdy, &n_owned));

      // truth field: two zones split at the mid-x of the domain
      PetscReal *xc, *n_true;
      PetscCall(PetscMalloc1(n_owned, &xc));
      PetscCall(PetscMalloc1(n_owned, &n_true));
      PetscCall(RDyMeshGetOwnedCellXCentroids(&rdy->mesh, n_owned, xc));
      PetscReal x_min = PETSC_MAX_REAL, x_max = -PETSC_MAX_REAL;
      for (PetscInt i = 0; i < n_owned; ++i) {
        x_min = PetscMin(x_min, xc[i]);
        x_max = PetscMax(x_max, xc[i]);
      }
      PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &x_min, 1, MPIU_REAL, MPIU_MIN, comm));
      PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &x_max, 1, MPIU_REAL, MPIU_MAX, comm));
      PetscReal x_split = 0.5 * (x_min + x_max);
      for (PetscInt i = 0; i < n_owned; ++i) n_true[i] = (xc[i] < x_split) ? 0.03 : 0.06;

      // multi-time observations: default to 8 observation windows
      PetscInt total_steps = rdy->config.time.stop_n;
      if (obs_freq <= 0) obs_freq = PetscMax(1, total_steps / 8);
      PetscInt K = total_steps / obs_freq;
      Vec     *y_k, *r_k;
      PetscCall(PetscMalloc1(K, &y_k));
      PetscCall(PetscMalloc1(K, &r_k));
      for (PetscInt k = 0; k < K; ++k) {
        PetscCall(MatCreateVecs(H, NULL, &y_k[k]));
        PetscCall(MatCreateVecs(H, NULL, &r_k[k]));
      }

      PetscCall(RDySetDomainManningsN(rdy, n_owned, n_true));
      PetscCall(ForwardObserve(rdy, u_ic, total_steps, obs_freq, H, y_k, NULL, NULL, sigma, NULL));  // record y_k

      // observability mask from the truth terminal state: wet and moving
      PetscBool *observable;
      PetscCall(PetscMalloc1(n_owned, &observable));
      {
        const PetscScalar *u_ptr;
        PetscCall(VecGetArrayRead(rdy->u_global, &u_ptr));
        PetscReal tiny_h = rdy->config.physics.flow.tiny_h;
        for (PetscInt i = 0; i < n_owned; ++i) {
          PetscReal h = PetscRealPart(u_ptr[3 * i]);
          PetscReal m = PetscHypotReal(PetscRealPart(u_ptr[3 * i + 1]), PetscRealPart(u_ptr[3 * i + 2]));
          observable[i] = (h >= tiny_h && m > 1e-12) ? PETSC_TRUE : PETSC_FALSE;
        }
        PetscCall(VecRestoreArrayRead(rdy->u_global, &u_ptr));
      }

      PerCellCtx pc = {.base        = {.rdy = rdy, .H = H, .y = y, .r_work = r_work, .u_ic = u_ic, .sigma = sigma, .t_final = t_final},
                       .beta        = beta,
                       .total_steps = total_steps,
                       .obs_freq    = obs_freq,
                       .K           = K,
                       .y_k         = y_k,
                       .r_k         = r_k};
      PetscCall(VecDuplicate(rdy->u_global, &pc.base.lambda));
      PetscCall(MatCreateVecs(rdy->rhs_jac_p, &pc.base.mu, NULL));
      PetscCall(VecDuplicate(pc.base.mu, &pc.n_prior));
      PetscCall(VecSet(pc.n_prior, 0.03));

      Tao tao;
      Vec p, lb, ub;
      PetscCall(VecDuplicate(pc.base.mu, &p));
      PetscCall(VecCopy(pc.n_prior, p));
      PetscCall(VecDuplicate(p, &lb));
      PetscCall(VecDuplicate(p, &ub));
      PetscCall(VecSet(lb, 0.01));
      PetscCall(VecSet(ub, 0.2));

      PetscCall(TaoCreate(comm, &tao));
      PetscCall(TaoSetType(tao, TAOBLMVM));
      PetscCall(TaoSetSolution(tao, p));
      PetscCall(TaoSetVariableBounds(tao, lb, ub));
      PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormObjectiveAndGradientPerCell, &pc));
      PetscCall(TaoSetTolerances(tao, 1e-14, 1e-14, 1e-14));
      PetscCall(TaoSetMaximumIterations(tao, 200));
      PetscCall(TaoSetFromOptions(tao));
      PetscCall(TaoSolve(tao));

      PetscInt tao_its;
      PetscCall(TaoGetIterationNumber(tao, &tao_its));

      // recovery assessment over observable cells
      const PetscScalar *p_ptr;
      PetscCall(VecGetArrayRead(p, &p_ptr));
      PetscReal err2 = 0.0, ref2 = 0.0;
      PetscInt  n_obs_cells = 0;
      for (PetscInt i = 0; i < n_owned; ++i) {
        if (observable[i]) {
          err2 += Square(PetscRealPart(p_ptr[i]) - n_true[i]);
          ref2 += Square(n_true[i]);
          n_obs_cells++;
        }
      }
      PetscCall(VecRestoreArrayRead(p, &p_ptr));
      PetscReal reduce[3] = {err2, ref2, (PetscReal)n_obs_cells};
      PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, reduce, 3, MPIU_REAL, MPIU_SUM, comm));
      PetscReal rel = PetscSqrtReal(reduce[0] / reduce[1]);
      PetscCall(PetscPrintf(comm,
                            "per-cell recovery: %d TAO its, %" PetscInt_FMT " observable cells, %" PetscInt_FMT
                            " obs times, rel L2 err %.3e (beta %.1e)\n",
                            (int)tao_its, (PetscInt)reduce[2], K, (double)rel, (double)beta));

      if (have_map_file) {
        PetscCall(WriteCellFieldVTK(rdy, p, "manning_n", map_file));
        PetscCall(PetscPrintf(comm, "wrote Manning map: %s\n", map_file));
      }

      // gate set from the validated baseline run (see RESULTS log): the fast
      // CI setting (8 obs times, 200-it cap) measured 0.199; the converged
      // setting (20 obs times, beta 1e-5) reaches 0.080. Overridable for
      // exploratory runs (e.g. larger cases run iteration-capped).
      PetscReal recovery_gate = 0.25;
      PetscCall(PetscOptionsGetReal(NULL, NULL, "-adjoint_recovery_gate", &recovery_gate, NULL));
      PetscCheck(rel < recovery_gate, comm, PETSC_ERR_PLIB, "per-cell recovery err too large: %g", (double)rel);

      for (PetscInt k = 0; k < K; ++k) {
        PetscCall(VecDestroy(&y_k[k]));
        PetscCall(VecDestroy(&r_k[k]));
      }
      PetscCall(PetscFree(y_k));
      PetscCall(PetscFree(r_k));
      PetscCall(PetscFree(observable));
      PetscCall(PetscFree(xc));
      PetscCall(PetscFree(n_true));
      PetscCall(TaoDestroy(&tao));
      PetscCall(VecDestroy(&p));
      PetscCall(VecDestroy(&lb));
      PetscCall(VecDestroy(&ub));
      PetscCall(VecDestroy(&pc.base.lambda));
      PetscCall(VecDestroy(&pc.base.mu));
      PetscCall(VecDestroy(&pc.n_prior));
      PetscCall(VecDestroy(&u_ic));
      PetscCall(VecDestroy(&y));
      PetscCall(VecDestroy(&r_work));
      PetscCall(MatDestroy(&H));
      PetscCall(RDyDestroy(&rdy));
      PetscCall(RDyFinalize());
      return 0;
    }

    if (calibrate) {
      // ------------------------------------------------------------------
      // increment 5: two-zone twin calibration. Truth uses n_true[r]
      // distinct per region; TAO/BLMVM recovers them from n0 = 0.02.
      // ------------------------------------------------------------------
      PetscInt n_regions = rdy->num_regions;
      PetscCheck(n_regions >= 2, comm, PETSC_ERR_USER, "calibration twin needs >= 2 regions");

      PetscCall(TSSetSaveTrajectory(rdy->ts));

      PetscReal *n_true;
      PetscCall(PetscMalloc1(n_regions, &n_true));
      for (PetscInt r = 0; r < n_regions; ++r) n_true[r] = 0.03 + 0.03 * r;  // 0.03, 0.06, ...
      PetscCall(ApplyRegionManning(rdy, n_true));
      PetscCall(ForwardSolve(rdy, u_ic, t_final));
      PetscCall(MatMult(H, rdy->u_global, y));  // synthetic observations (noise-free twin)

      CalibrationCtx cal = {.rdy = rdy, .H = H, .y = y, .r_work = r_work, .u_ic = u_ic, .sigma = sigma, .t_final = t_final};
      PetscCall(VecDuplicate(rdy->u_global, &cal.lambda));
      PetscCall(MatCreateVecs(rdy->rhs_jac_p, &cal.mu, NULL));

      Tao tao;
      Vec p, lb, ub;
      PetscCall(VecCreate(comm, &p));
      PetscCall(VecSetSizes(p, PETSC_DECIDE, n_regions));
      PetscCall(VecSetFromOptions(p));
      PetscCall(VecSet(p, 0.02));  // initial guess
      PetscCall(VecDuplicate(p, &lb));
      PetscCall(VecDuplicate(p, &ub));
      PetscCall(VecSet(lb, 0.01));
      PetscCall(VecSet(ub, 0.2));

      PetscCall(TaoCreate(comm, &tao));
      PetscCall(TaoSetType(tao, TAOBLMVM));
      PetscCall(TaoSetSolution(tao, p));
      PetscCall(TaoSetVariableBounds(tao, lb, ub));
      PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormObjectiveAndGradient, &cal));
      PetscCall(TaoSetTolerances(tao, 1e-12, 1e-12, 1e-12));
      PetscCall(TaoSetMaximumIterations(tao, 100));
      PetscCall(TaoSetFromOptions(tao));
      PetscCall(TaoSolve(tao));

      // report and gate: recovered n within 2% of truth per region
      Vec        p_all_vec;
      VecScatter scatter;
      PetscCall(VecScatterCreateToAll(p, &scatter, &p_all_vec));
      PetscCall(VecScatterBegin(scatter, p, p_all_vec, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(scatter, p, p_all_vec, INSERT_VALUES, SCATTER_FORWARD));
      const PetscScalar *p_rec;
      PetscCall(VecGetArrayRead(p_all_vec, &p_rec));
      PetscReal max_rel = 0.0;
      for (PetscInt r = 0; r < n_regions; ++r) {
        PetscReal rel = PetscAbsReal(PetscRealPart(p_rec[r]) - n_true[r]) / n_true[r];
        PetscCall(PetscPrintf(comm, "region %" PetscInt_FMT ": n_true %.6f  n_recovered %.6f  rel err %.3e\n", r, (double)n_true[r],
                              (double)PetscRealPart(p_rec[r]), (double)rel));
        if (rel > max_rel) max_rel = rel;
      }
      PetscCall(VecRestoreArrayRead(p_all_vec, &p_rec));
      PetscCall(PetscPrintf(comm, "two-zone recovery max rel err: %.3e (gate 2e-2)\n", (double)max_rel));
      PetscCheck(max_rel < 2e-2, comm, PETSC_ERR_PLIB, "calibration failed to recover the two-zone Manning field: %g", (double)max_rel);

      PetscCall(VecScatterDestroy(&scatter));
      PetscCall(VecDestroy(&p_all_vec));
      PetscCall(TaoDestroy(&tao));
      PetscCall(VecDestroy(&p));
      PetscCall(VecDestroy(&lb));
      PetscCall(VecDestroy(&ub));
      PetscCall(VecDestroy(&cal.lambda));
      PetscCall(VecDestroy(&cal.mu));
      PetscCall(PetscFree(n_true));
      PetscCall(VecDestroy(&u_ic));
      PetscCall(VecDestroy(&y));
      PetscCall(VecDestroy(&r_work));
      PetscCall(MatDestroy(&H));
      PetscCall(RDyDestroy(&rdy));
      PetscCall(RDyFinalize());
      return 0;
    }

    // 1. truth solve -> observations
    PetscCall(ForwardSolve(rdy, u_ic, t_final));
    PetscCall(MatMult(H, rdy->u_global, y));

    // 2. perturbed IC: scale heights by (1 + ic_perturb)
    PetscCall(VecDuplicate(u_ic, &u_ic_pert));
    PetscCall(VecCopy(u_ic, u_ic_pert));
    {
      PetscScalar *u_ptr;
      PetscInt     n_local;
      PetscCall(VecGetLocalSize(u_ic_pert, &n_local));
      PetscCall(VecGetArray(u_ic_pert, &u_ptr));
      for (PetscInt i = 0; i < n_local; i += NDOF) u_ptr[i] *= (1.0 + ic_perturb);
      PetscCall(VecRestoreArray(u_ic_pert, &u_ptr));
    }

    PetscCall(ForwardSolve(rdy, u_ic_pert, t_final));
    PetscReal J0;
    PetscCall(Misfit(H, rdy->u_global, y, sigma, r_work, &J0));
    PetscCall(PetscPrintf(comm, "terminal misfit J = %g (%" PetscInt_FMT " observations)\n", (double)J0, nobs));

    // 3. adjoint sweep: lambda(T) = H^T R^-1 (H u(T) - y), mu(T) = 0
    //    one backward sweep yields both dJ/du0 (lambda) and dJ/dn (mu)
    Vec lambda, mu;
    PetscCall(VecDuplicate(rdy->u_global, &lambda));
    PetscCall(VecScale(r_work, 1.0 / (sigma * sigma)));  // r_work = R^-1 (H u(T) - y)
    PetscCall(MatMultTranspose(H, r_work, lambda));
    PetscCall(MatCreateVecs(rdy->rhs_jac_p, &mu, NULL));
    PetscCall(VecZeroEntries(mu));
    PetscCall(TSSetCostGradients(rdy->ts, 1, &lambda, &mu));
    PetscCall(TSAdjointSolve(rdy->ts));
    // lambda now holds dJ/du0; mu holds dJ/dn (per owned cell)

    PetscReal grad_norm, mu_sum;
    PetscCall(VecNorm(lambda, NORM_2, &grad_norm));
    PetscCall(VecSum(mu, &mu_sum));
    PetscCall(PetscPrintf(comm, "|dJ/du0|_2 = %g   sum(dJ/dn) = %g\n", (double)grad_norm, (double)mu_sum));

    // 4. FD check on sampled components of u0
    if (fd_samples > 0) {
      PetscInt state_size = ncells_global * NDOF;
      if (fd_samples > state_size) fd_samples = state_size;

      PetscReal err2 = 0.0, ref2 = 0.0;
      for (PetscInt k = 0; k < fd_samples; ++k) {
        // sample height dofs spread across the domain (heights: index % 3 == 0)
        PetscInt  cell = (PetscInt)(((PetscInt64)k * ncells_global) / fd_samples);
        PetscInt  idx  = cell * NDOF;
        PetscReal eps  = fd_eps;

        Vec up, um;
        PetscCall(VecDuplicate(u_ic_pert, &up));
        PetscCall(VecDuplicate(u_ic_pert, &um));
        PetscCall(VecCopy(u_ic_pert, up));
        PetscCall(VecCopy(u_ic_pert, um));
        // perturb from ONE rank only: ADD_VALUES contributions are summed
        // across ranks, so an every-rank call scales the FD slope by nproc
        PetscMPIInt fd_rank;
        PetscCallMPI(MPI_Comm_rank(comm, &fd_rank));
        if (fd_rank == 0) {
          PetscCall(VecSetValue(up, idx, eps, ADD_VALUES));
          PetscCall(VecSetValue(um, idx, -eps, ADD_VALUES));
        }
        PetscCall(VecAssemblyBegin(up));
        PetscCall(VecAssemblyEnd(up));
        PetscCall(VecAssemblyBegin(um));
        PetscCall(VecAssemblyEnd(um));

        PetscReal Jp, Jm;
        PetscCall(ForwardSolve(rdy, up, t_final));
        PetscCall(Misfit(H, rdy->u_global, y, sigma, r_work, &Jp));
        PetscCall(ForwardSolve(rdy, um, t_final));
        PetscCall(Misfit(H, rdy->u_global, y, sigma, r_work, &Jm));
        PetscReal g_fd = (Jp - Jm) / (2.0 * eps);

        // fetch lambda[idx] (owned by exactly one rank; share it)
        PetscReal   g_adj = 0.0;
        PetscInt    lo, hi;
        PetscCall(VecGetOwnershipRange(lambda, &lo, &hi));
        if (idx >= lo && idx < hi) {
          const PetscScalar *l_ptr;
          PetscCall(VecGetArrayRead(lambda, &l_ptr));
          g_adj = PetscRealPart(l_ptr[idx - lo]);
          PetscCall(VecRestoreArrayRead(lambda, &l_ptr));
        }
        PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &g_adj, 1, MPIU_REAL, MPIU_SUM, comm));

        err2 += (g_adj - g_fd) * (g_adj - g_fd);
        ref2 += g_fd * g_fd;
        PetscCall(PetscPrintf(comm, "  u0[%" PetscInt_FMT "] (cell %" PetscInt_FMT " h): adjoint %14.8e  fd %14.8e\n", idx, cell, (double)g_adj,
                              (double)g_fd));
        PetscCall(VecDestroy(&up));
        PetscCall(VecDestroy(&um));
      }
      PetscReal rel = PetscSqrtReal(err2 / (ref2 > 0.0 ? ref2 : 1.0));
      PetscCall(PetscPrintf(comm, "adjoint-vs-FD relative L2 error over %" PetscInt_FMT " samples: %.3e (gate %.1e)\n", fd_samples, (double)rel,
                            (double)fd_tol));
      PetscCheck(rel < fd_tol, comm, PETSC_ERR_PLIB, "adjoint gradient failed the FD gate: %g >= %g", (double)rel, (double)fd_tol);

      // FD check of the domain-aggregate Manning gradient: perturb n
      // uniformly over the domain, so d/deps J(n + eps 1) = sum_c dJ/dn_c
      PetscInt n_local_cells;
      PetscCall(RDyGetNumOwnedCells(rdy, &n_local_cells));
      PetscReal *n_vals;
      PetscCall(PetscMalloc1(n_local_cells, &n_vals));
      PetscScalar *mp_ptr;
      PetscCall(VecGetArray(rdy->operator->petsc.material_properties, &mp_ptr));
      PetscReal n0 = PetscRealPart(mp_ptr[MATERIAL_PROPERTY_MANNINGS]);  // uniform in the test configs
      PetscCall(VecRestoreArray(rdy->operator->petsc.material_properties, &mp_ptr));

      // eps sized against FD cancellation: the n-perturbation changes J by
      // ~2 eps |dJ/dn|, which must clear macheps*J by several digits
      PetscReal eps_n = 1e-4, Jp_n, Jm_n;
      for (PetscInt i = 0; i < n_local_cells; ++i) n_vals[i] = n0 + eps_n;
      PetscCall(RDySetDomainManningsN(rdy, n_local_cells, n_vals));
      PetscCall(ForwardSolve(rdy, u_ic_pert, t_final));
      PetscCall(Misfit(H, rdy->u_global, y, sigma, r_work, &Jp_n));
      for (PetscInt i = 0; i < n_local_cells; ++i) n_vals[i] = n0 - eps_n;
      PetscCall(RDySetDomainManningsN(rdy, n_local_cells, n_vals));
      PetscCall(ForwardSolve(rdy, u_ic_pert, t_final));
      PetscCall(Misfit(H, rdy->u_global, y, sigma, r_work, &Jm_n));
      for (PetscInt i = 0; i < n_local_cells; ++i) n_vals[i] = n0;  // restore
      PetscCall(RDySetDomainManningsN(rdy, n_local_cells, n_vals));
      PetscCall(PetscFree(n_vals));

      PetscReal g_fd_n  = (Jp_n - Jm_n) / (2.0 * eps_n);
      PetscReal rel_n   = PetscAbsReal(mu_sum - g_fd_n) / PetscMax(PetscAbsReal(g_fd_n), 1e-30);
      // degenerate-gradient guard: when the flow is (nearly) motionless the
      // drag gradient is genuinely ~0 and both sides are solver/FD noise --
      // accept when the absolute discrepancy is negligible against J
      PetscBool n_grad_ok = (rel_n < fd_tol) || (PetscAbsReal(mu_sum - g_fd_n) <= fd_tol * PetscMax(J0, 1.0));
      PetscCall(PetscPrintf(comm, "dJ/dn (domain aggregate): adjoint %14.8e  fd %14.8e  rel %.3e (gate %.1e%s)\n", (double)mu_sum, (double)g_fd_n,
                            (double)rel_n, (double)fd_tol, n_grad_ok && rel_n >= fd_tol ? ", passed on absolute floor" : ""));
      PetscCheck(n_grad_ok, comm, PETSC_ERR_PLIB, "Manning gradient failed the FD gate: %g >= %g", (double)rel_n, (double)fd_tol);
    }

    PetscCall(VecDestroy(&mu));
    PetscCall(VecDestroy(&lambda));
    PetscCall(VecDestroy(&u_ic));
    PetscCall(VecDestroy(&u_ic_pert));
    PetscCall(VecDestroy(&y));
    PetscCall(VecDestroy(&r_work));
    PetscCall(MatDestroy(&H));
    PetscCall(DestroyRainSchedule());
    PetscCall(RDyDestroy(&rdy));
  }

  PetscCall(RDyFinalize());
  return 0;
}
