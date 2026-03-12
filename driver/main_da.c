#include <petscmat.h>
#include <petscsys.h>
#include <petscvec.h>
#include <rdycore.h>
#include <stdio.h>

#include "petscda.h"

static const char *help_str =
    "rdycore_da - a standalone LETKF driver for RDycore\n"
    "usage: rdycore_da [options] <input.yaml>\n\n"
    "This first-cut driver supports synthetic height observations generated from a\n"
    "truth RDycore run and a LETKF analysis over a sequentially advanced ensemble.\n";

typedef struct {
  char      config_file[PETSC_MAX_PATH_LEN];
  char      output_prefix[PETSC_MAX_PATH_LEN];
  PetscInt  ensemble_size;
  PetscInt  obs_frequency;
  PetscInt  obs_stride;
  PetscInt  progress_frequency;
  PetscInt  random_seed;
  PetscInt  num_observations_vertex;
  PetscReal obs_error;
  PetscBool use_global_localization;
} DAOptions;

typedef struct {
  DAOptions   options;
  RDyTimeUnit time_unit;
  PetscDA     da;
  RDy         truth_rdy;
  RDy        *member_rdys;
  PetscRandom rng;
  Mat         H;
  Mat         H1;
  Mat         Q;
  Vec         truth_state;
  Vec         truth_obs;
  Vec         observation;
  Vec         obs_noise;
  Vec         obs_error_var;
  Vec         x_mean;
  Vec         member_work;
  Vec         rmse_work;
  PetscInt    ndof;
  PetscInt    state_size;
  PetscInt    obs_size;
  PetscInt    obs_count;
  FILE       *diag_fp;
} DARun;

static void usage(const char *exe_name) {
  fprintf(stderr, "%s: usage:\n", exe_name);
  fprintf(stderr, "%s [options] <input.yaml>\n\n", exe_name);
}

static PetscErrorCode DAOptionsSetDefaults(DAOptions *options) {
  PetscFunctionBeginUser;

  options->config_file[0]          = '\0';
  options->output_prefix[0]        = '\0';
  options->ensemble_size           = 8;
  options->obs_frequency           = 1;
  options->obs_stride              = 4;
  options->progress_frequency      = 1;
  options->random_seed             = 12345;
  options->num_observations_vertex = 7;
  options->obs_error               = 1e-2;
  options->use_global_localization = PETSC_FALSE;

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DAOptionsParse(DAOptions *options, int argc, char **argv) {
  char      bool_value[32] = {0};
  PetscBool has_bool = PETSC_FALSE, has_value = PETSC_FALSE;

  PetscFunctionBeginUser;

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "RDycore LETKF Driver Options", NULL);
  PetscCall(PetscOptionsInt("-ensemble_size", "Number of ensemble members", "", options->ensemble_size, &options->ensemble_size, NULL));
  PetscCall(PetscOptionsInt("-obs_frequency", "Assimilation frequency in RDyAdvance calls", "", options->obs_frequency, &options->obs_frequency, NULL));
  PetscCall(PetscOptionsInt("-obs_stride", "Subsampling stride for synthetic height observations", "", options->obs_stride, &options->obs_stride, NULL));
  PetscCall(PetscOptionsInt("-progress_frequency", "Print progress every N RDyAdvance calls", "", options->progress_frequency, &options->progress_frequency, NULL));
  PetscCall(PetscOptionsReal("-obs_error", "Observation noise standard deviation", "", options->obs_error, &options->obs_error, NULL));
  PetscCall(PetscOptionsInt("-random_seed", "Random seed for ensemble and observation noise", "", options->random_seed, &options->random_seed, NULL));
  PetscCall(PetscOptionsInt("-num_observations_vertex", "Number of observations per state location for localization", "", options->num_observations_vertex, &options->num_observations_vertex, NULL));
  PetscCall(PetscOptionsString("-output_prefix", "Prefix for future DA diagnostics output", "", options->output_prefix, options->output_prefix, sizeof(options->output_prefix), NULL));
  PetscOptionsEnd();

  PetscCall(PetscOptionsHasName(NULL, NULL, "-use_global_localization", &has_bool));
  if (has_bool) {
    PetscCall(PetscOptionsGetString(NULL, NULL, "-use_global_localization", bool_value, sizeof(bool_value), &has_value));
    if (!has_value || (options->config_file[0] && !strcmp(bool_value, options->config_file))) {
      options->use_global_localization = PETSC_TRUE;
    } else if (!strcmp(bool_value, "1") || !strcmp(bool_value, "true") || !strcmp(bool_value, "yes")) {
      options->use_global_localization = PETSC_TRUE;
    } else if (!strcmp(bool_value, "0") || !strcmp(bool_value, "false") || !strcmp(bool_value, "no")) {
      options->use_global_localization = PETSC_FALSE;
    } else {
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Unsupported value '%s' for -use_global_localization", bool_value);
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DAOptionsValidate(const DAOptions *options) {
  PetscFunctionBeginUser;

  PetscCheck(options->config_file[0], PETSC_COMM_WORLD, PETSC_ERR_USER, "A RDycore YAML input file is required.");
  PetscCheck(options->ensemble_size > 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "-ensemble_size must be at least 2.");
  PetscCheck(options->obs_frequency > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "-obs_frequency must be positive.");
  PetscCheck(options->obs_stride > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "-obs_stride must be positive.");
  PetscCheck(options->progress_frequency >= 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "-progress_frequency must be non-negative.");
  PetscCheck(options->obs_error > 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "-obs_error must be positive.");
  PetscCheck(options->num_observations_vertex > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "-num_observations_vertex must be positive.");

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DAOptionsPrint(const DAOptions *options) {
  PetscFunctionBeginUser;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "RDycore LETKF driver scope\n"
                        "========================\n"
                        "  Config file             : %s\n"
                        "  Ensemble size           : %" PetscInt_FMT "\n"
                        "  Observation frequency   : %" PetscInt_FMT "\n"
                        "  Observation stride      : %" PetscInt_FMT "\n"
                        "  Progress frequency      : %" PetscInt_FMT "\n"
                        "  Observation noise std   : %.6g\n"
                        "  Random seed             : %" PetscInt_FMT "\n"
                        "  Obs per vertex          : %" PetscInt_FMT "\n"
                        "  Global localization     : %s\n"
                        "  Output prefix           : %s\n"
                        "\n"
                        "First-cut scope\n"
                        "  - synthetic observations only\n"
                        "  - no rainfall dataset options\n"
                        "  - no boundary-condition dataset options\n"
                        "  - one RDycore model instance per ensemble member\n\n",
                        options->config_file, options->ensemble_size, options->obs_frequency, options->obs_stride, options->progress_frequency,
                        (double)options->obs_error,
                        options->random_seed, options->num_observations_vertex, options->use_global_localization ? "yes" : "no",
                        options->output_prefix[0] ? options->output_prefix : "(none)"));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeRMSE(Vec v1, Vec v2, Vec work, PetscInt n, PetscReal *rmse) {
  PetscReal norm;

  PetscFunctionBeginUser;
  PetscCall(VecWAXPY(work, -1.0, v2, v1));
  PetscCall(VecNorm(work, NORM_2, &norm));
  *rmse = norm / PetscSqrtReal((PetscReal)n);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateTruthAndMemberModels(DARun *da_run) {
  PetscFunctionBeginUser;

  PetscCall(RDyCreate(PETSC_COMM_WORLD, da_run->options.config_file, &da_run->truth_rdy));
  PetscCall(RDySetLogFile(da_run->truth_rdy, "/dev/null"));
  PetscCall(RDySetup(da_run->truth_rdy));
  PetscCall(RDyDisableOutput(da_run->truth_rdy));
  PetscCall(RDyGetTimeUnit(da_run->truth_rdy, &da_run->time_unit));

  PetscCall(PetscCalloc1(da_run->options.ensemble_size, &da_run->member_rdys));
  for (PetscInt i = 0; i < da_run->options.ensemble_size; ++i) {
    PetscCall(RDyCreate(PETSC_COMM_WORLD, da_run->options.config_file, &da_run->member_rdys[i]));
    PetscCall(RDySetLogFile(da_run->member_rdys[i], "/dev/null"));
    PetscCall(RDySetup(da_run->member_rdys[i]));
    PetscCall(RDyDisableOutput(da_run->member_rdys[i]));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateObservationMatrices(RDy rdy, PetscInt ndof, PetscInt obs_stride, Mat *H, Mat *H1, PetscInt *nobs_out) {
  PetscInt num_global_cells, nobs, rstart, rend;

  PetscFunctionBeginUser;

  PetscCall(RDyGetNumGlobalCells(rdy, &num_global_cells));
  nobs = (num_global_cells + obs_stride - 1) / obs_stride;

  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, nobs, num_global_cells * ndof, 1, NULL, 1, NULL, H));
  PetscCall(MatSetFromOptions(*H));
  PetscCall(MatSetUp(*H));

  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, nobs, num_global_cells, 1, NULL, 1, NULL, H1));
  PetscCall(MatSetFromOptions(*H1));
  PetscCall(MatSetUp(*H1));

  PetscCall(MatGetOwnershipRange(*H, &rstart, &rend));
  for (PetscInt obs_idx = rstart; obs_idx < rend; ++obs_idx) {
    PetscInt cell_id = obs_idx * obs_stride;

    if (cell_id < num_global_cells) {
      PetscCall(MatSetValue(*H1, obs_idx, cell_id, 1.0, INSERT_VALUES));
      PetscCall(MatSetValue(*H, obs_idx, cell_id * ndof, 1.0, INSERT_VALUES));
    }
  }

  PetscCall(MatAssemblyBegin(*H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(*H1, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*H1, MAT_FINAL_ASSEMBLY));

  *nobs_out = nobs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateGlobalCoordinateVector(PetscInt global_size, PetscInt local_size, const PetscInt *indices, const PetscReal *values, Vec *vec) {
  PetscFunctionBeginUser;

  PetscCall(VecCreate(PETSC_COMM_WORLD, vec));
  PetscCall(VecSetSizes(*vec, local_size, global_size));
  PetscCall(VecSetFromOptions(*vec));
  PetscCall(VecSet(*vec, 0.0));
  for (PetscInt i = 0; i < local_size; ++i) PetscCall(VecSetValue(*vec, indices[i], values[i], INSERT_VALUES));
  PetscCall(VecAssemblyBegin(*vec));
  PetscCall(VecAssemblyEnd(*vec));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateGlobalLocalizationMatrix(PetscInt num_global_cells, PetscInt nobs, Mat *Q) {
  PetscInt rstart, rend;

  PetscFunctionBeginUser;

  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_global_cells, nobs, nobs, NULL, 0, NULL, Q));
  PetscCall(MatSetFromOptions(*Q));
  PetscCall(MatSetUp(*Q));

  PetscCall(MatGetOwnershipRange(*Q, &rstart, &rend));
  for (PetscInt row = rstart; row < rend; ++row) {
    for (PetscInt col = 0; col < nobs; ++col) PetscCall(MatSetValue(*Q, row, col, 1.0, INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(*Q, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*Q, MAT_FINAL_ASSEMBLY));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateLocalizationMatrix(RDy rdy, const DAOptions *options, PetscDA da, Mat H1, PetscInt nobs, Mat *Q) {
  PetscInt  num_global_cells, num_owned_cells;
  PetscInt *natural_ids;
  PetscReal *x, *y, *z;
  PetscBool is_letkf;

  PetscFunctionBeginUser;

  PetscCall(PetscObjectTypeCompare((PetscObject)da, PETSCDALETKF, &is_letkf));
  PetscCheck(is_letkf, PETSC_COMM_WORLD, PETSC_ERR_SUP, "rdycore_da currently supports only PETSCDALETKF.");

  PetscCall(RDyGetNumGlobalCells(rdy, &num_global_cells));
  if (options->use_global_localization) {
    PetscCall(CreateGlobalLocalizationMatrix(num_global_cells, nobs, Q));
    PetscCall(PetscDALETKFSetObsPerVertex(da, nobs));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(CreateGlobalLocalizationMatrix(num_global_cells, nobs, Q));
  PetscCall(PetscDALETKFSetObsPerVertex(da, nobs));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode OpenDiagnosticsFile(DARun *da_run) {
  char file_name[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;

  da_run->diag_fp = NULL;
  if (!da_run->options.output_prefix[0]) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscSNPrintf(file_name, sizeof(file_name), "%s.da.dat", da_run->options.output_prefix));
  PetscCall(PetscFOpen(PETSC_COMM_WORLD, file_name, "w", &da_run->diag_fp));
  PetscCall(PetscFPrintf(PETSC_COMM_WORLD, da_run->diag_fp, "# step time rmse_forecast rmse_analysis assimilated\n"));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode WriteDiagnostics(DARun *da_run, PetscInt step, PetscReal time, PetscReal rmse_forecast, PetscReal rmse_analysis, PetscBool assimilated) {
  PetscFunctionBeginUser;

  if (da_run->diag_fp) {
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, da_run->diag_fp, "%" PetscInt_FMT " %.12e %.12e %.12e %d\n", step, (double)time,
                           (double)rmse_forecast, (double)rmse_analysis, (int)assimilated));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SyncEnsembleIntoModels(DARun *da_run) {
  Vec member_state = NULL;

  PetscFunctionBeginUser;

  for (PetscInt i = 0; i < da_run->options.ensemble_size; ++i) {
    PetscCall(PetscDAGetEnsembleMember(da_run->da, i, &member_state));
    PetscCall(RDySetInitialConditions(da_run->member_rdys[i], member_state));
    PetscCall(PetscDARestoreEnsembleMember(da_run->da, i, &member_state));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AdvanceEnsemble(DARun *da_run) {
  Vec member_state = NULL;

  PetscFunctionBeginUser;

  for (PetscInt i = 0; i < da_run->options.ensemble_size; ++i) {
    PetscCall(RDyAdvance(da_run->member_rdys[i]));
    PetscCall(RDyCopySolution(da_run->member_rdys[i], da_run->member_work));
    PetscCall(PetscDASetEnsembleMember(da_run->da, i, da_run->member_work));
    PetscCall(PetscDAGetEnsembleMember(da_run->da, i, &member_state));
    PetscCall(RDySetInitialConditions(da_run->member_rdys[i], member_state));
    PetscCall(PetscDARestoreEnsembleMember(da_run->da, i, &member_state));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ConfigureDA(DARun *da_run) {
  PetscInt  local_state_size, local_obs_size;
  PetscBool is_letkf;

  PetscFunctionBeginUser;

  PetscCall(RDyCreatePrognosticVec(da_run->truth_rdy, &da_run->truth_state));
  PetscCall(RDyCopySolution(da_run->truth_rdy, da_run->truth_state));
  PetscCall(VecDuplicate(da_run->truth_state, &da_run->x_mean));
  PetscCall(VecDuplicate(da_run->truth_state, &da_run->member_work));
  PetscCall(VecDuplicate(da_run->truth_state, &da_run->rmse_work));

  PetscCall(VecGetBlockSize(da_run->truth_state, &da_run->ndof));
  PetscCall(VecGetSize(da_run->truth_state, &da_run->state_size));
  PetscCheck(da_run->ndof == 3, PETSC_COMM_WORLD, PETSC_ERR_SUP, "rdycore_da currently expects a 3-DOF SWE state, got block size %" PetscInt_FMT,
             da_run->ndof);

  PetscCall(CreateObservationMatrices(da_run->truth_rdy, da_run->ndof, da_run->options.obs_stride, &da_run->H, &da_run->H1, &da_run->obs_size));
  PetscCall(MatCreateVecs(da_run->H, NULL, &da_run->observation));
  PetscCall(VecDuplicate(da_run->observation, &da_run->truth_obs));
  PetscCall(VecDuplicate(da_run->observation, &da_run->obs_noise));
  PetscCall(VecDuplicate(da_run->observation, &da_run->obs_error_var));
  PetscCall(VecSet(da_run->obs_error_var, da_run->options.obs_error * da_run->options.obs_error));

  PetscCall(PetscDACreate(PETSC_COMM_WORLD, &da_run->da));
  PetscCall(PetscDASetType(da_run->da, PETSCDALETKF));
  PetscCall(PetscDASetSizes(da_run->da, da_run->state_size, da_run->obs_size, da_run->options.ensemble_size));
  PetscCall(VecGetLocalSize(da_run->truth_state, &local_state_size));
  PetscCall(VecGetLocalSize(da_run->observation, &local_obs_size));
  PetscCall(PetscDASetLocalSizes(da_run->da, local_state_size, local_obs_size));
  PetscCall(PetscDASetNDOF(da_run->da, da_run->ndof));
  PetscCall(PetscDASetFromOptions(da_run->da));
  PetscCall(PetscDASetUp(da_run->da));
  PetscCall(PetscObjectTypeCompare((PetscObject)da_run->da, PETSCDALETKF, &is_letkf));
  PetscCheck(is_letkf, PETSC_COMM_WORLD, PETSC_ERR_SUP, "rdycore_da currently supports only the LETKF PetscDA type.");

  PetscCall(PetscDASetObsErrorVariance(da_run->da, da_run->obs_error_var));
  PetscCall(CreateLocalizationMatrix(da_run->truth_rdy, &da_run->options, da_run->da, da_run->H1, da_run->obs_size, &da_run->Q));
  PetscCall(PetscDALETKFSetLocalization(da_run->da, da_run->Q, da_run->H));

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &da_run->rng));
  {
    PetscMPIInt rank;

    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    PetscCall(PetscRandomSetSeed(da_run->rng, (unsigned long)(da_run->options.random_seed + rank)));
  }
  PetscCall(PetscRandomSetFromOptions(da_run->rng));
  PetscCall(PetscRandomSeed(da_run->rng));

  PetscCall(InitializeEnsemble(da_run->da, da_run->truth_state, da_run->options.ensemble_size, da_run->options.obs_error, da_run->rng));
  PetscCall(SyncEnsembleIntoModels(da_run));
  PetscCall(OpenDiagnosticsFile(da_run));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrintProgress(const DARun *da_run, PetscInt step, PetscReal time, PetscReal rmse_forecast, PetscReal rmse_analysis) {
  PetscFunctionBeginUser;

  if (da_run->options.progress_frequency == 0) PetscFunctionReturn(PETSC_SUCCESS);
  if ((step % da_run->options.progress_frequency) == 0) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step %4" PetscInt_FMT ", time %10.6f  RMSE_forecast %.6e  RMSE_analysis %.6e\n", step,
                          (double)time, (double)rmse_forecast, (double)rmse_analysis));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RunDataAssimilation(DARun *da_run) {
  PetscReal sum_rmse_forecast = 0.0, sum_rmse_analysis = 0.0;
  PetscInt  num_stats = 0;

  PetscFunctionBeginUser;

  PetscCall(PetscDAComputeEnsembleMean(da_run->da, da_run->x_mean));
  {
    PetscReal rmse_initial;

    PetscCall(ComputeRMSE(da_run->x_mean, da_run->truth_state, da_run->rmse_work, da_run->state_size, &rmse_initial));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Initial RMSE %.6e\n", (double)rmse_initial));
    PetscCall(WriteDiagnostics(da_run, 0, 0.0, rmse_initial, rmse_initial, PETSC_FALSE));
  }

  while (!RDyFinished(da_run->truth_rdy)) {
    PetscInt  step;
    PetscReal time, rmse_forecast, rmse_analysis;
    PetscBool assimilated = PETSC_FALSE;

    PetscCall(AdvanceEnsemble(da_run));
    PetscCall(RDyAdvance(da_run->truth_rdy));
    PetscCall(RDyCopySolution(da_run->truth_rdy, da_run->truth_state));
    PetscCall(RDyGetStep(da_run->truth_rdy, &step));
    PetscCall(RDyGetTime(da_run->truth_rdy, da_run->time_unit, &time));

    PetscCall(PetscDAComputeEnsembleMean(da_run->da, da_run->x_mean));
    PetscCall(ComputeRMSE(da_run->x_mean, da_run->truth_state, da_run->rmse_work, da_run->state_size, &rmse_forecast));
    rmse_analysis = rmse_forecast;

    if ((step % da_run->options.obs_frequency) == 0) {
      assimilated = PETSC_TRUE;
      PetscCall(MatMult(da_run->H, da_run->truth_state, da_run->truth_obs));
      PetscCall(VecSetRandomGaussian(da_run->obs_noise, da_run->rng, 0.0, da_run->options.obs_error));
      PetscCall(VecWAXPY(da_run->observation, 1.0, da_run->obs_noise, da_run->truth_obs));
      PetscCall(PetscDAAnalysis(da_run->da, da_run->observation, da_run->H));
      PetscCall(PetscDAComputeEnsembleMean(da_run->da, da_run->x_mean));
      PetscCall(ComputeRMSE(da_run->x_mean, da_run->truth_state, da_run->rmse_work, da_run->state_size, &rmse_analysis));
      PetscCall(SyncEnsembleIntoModels(da_run));
      da_run->obs_count++;
    }

    sum_rmse_forecast += rmse_forecast;
    sum_rmse_analysis += rmse_analysis;
    num_stats++;

    PetscCall(PrintProgress(da_run, step, time, rmse_forecast, rmse_analysis));
    PetscCall(WriteDiagnostics(da_run, step, time, rmse_forecast, rmse_analysis, assimilated));
  }

  if (num_stats > 0) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "\nDA summary\n"
                          "  Forecast RMSE average : %.6e\n"
                          "  Analysis RMSE average : %.6e\n"
                          "  Analyses performed    : %" PetscInt_FMT "\n",
                          (double)(sum_rmse_forecast / num_stats), (double)(sum_rmse_analysis / num_stats), da_run->obs_count));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroyDARun(DARun *da_run) {
  PetscFunctionBeginUser;

  if (da_run->diag_fp) PetscCall(PetscFClose(PETSC_COMM_WORLD, da_run->diag_fp));
  PetscCall(PetscRandomDestroy(&da_run->rng));
  PetscCall(MatDestroy(&da_run->H));
  PetscCall(MatDestroy(&da_run->H1));
  PetscCall(MatDestroy(&da_run->Q));
  PetscCall(VecDestroy(&da_run->truth_state));
  PetscCall(VecDestroy(&da_run->truth_obs));
  PetscCall(VecDestroy(&da_run->observation));
  PetscCall(VecDestroy(&da_run->obs_noise));
  PetscCall(VecDestroy(&da_run->obs_error_var));
  PetscCall(VecDestroy(&da_run->x_mean));
  PetscCall(VecDestroy(&da_run->member_work));
  PetscCall(VecDestroy(&da_run->rmse_work));
  PetscCall(PetscDADestroy(&da_run->da));

  if (da_run->member_rdys) {
    for (PetscInt i = 0; i < da_run->options.ensemble_size; ++i) PetscCall(RDyDestroy(&da_run->member_rdys[i]));
    PetscCall(PetscFree(da_run->member_rdys));
  }
  PetscCall(RDyDestroy(&da_run->truth_rdy));

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[]) {
  DARun da_run = {0};

  if (argc < 2) {
    usage(argv[0]);
    return 0;
  }

  PetscCall(DAOptionsSetDefaults(&da_run.options));
  if (argv[argc - 1][0] != '-') {
    strncpy(da_run.options.config_file, argv[argc - 1], sizeof(da_run.options.config_file) - 1);
    da_run.options.config_file[sizeof(da_run.options.config_file) - 1] = '\0';
  }

  PetscCall(RDyInit(argc, argv, help_str));

  const char *rdy_build_config = NULL;
  PetscCall(RDyGetBuildConfiguration(&rdy_build_config));
  PetscCall(PetscFPrintf(PETSC_COMM_WORLD, stderr, "%s", rdy_build_config));

  if (strcmp(argv[1], "-help")) {
    PetscCall(DAOptionsParse(&da_run.options, argc, argv));
    PetscCall(DAOptionsValidate(&da_run.options));
    PetscCall(DAOptionsPrint(&da_run.options));

    PetscCall(CreateTruthAndMemberModels(&da_run));
    PetscCall(ConfigureDA(&da_run));
    PetscCall(RunDataAssimilation(&da_run));
    PetscCall(DestroyDARun(&da_run));
  }

  PetscCall(RDyFinalize());
  return 0;
}