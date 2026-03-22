#include <petscsys.h>
#include <petscda.h>
#include <rdycore.h>
#include <private/rdycoreimpl.h>

static const char *help_str =
    "rdycore_letkf - LETKF data assimilation twin experiment for RDycore\n"
    "usage: rdycore_letkf [options] <filename.yaml>\n\n"
    "Runs a twin experiment: a 'truth' RDycore simulation generates artificial\n"
    "observations, and an ensemble of RDycore forecasts is corrected using the\n"
    "PetscDA LETKF (Local Ensemble Transform Kalman Filter).\n\n"
    "Options:\n"
    "  -letkf_ensemble_size <int>   Number of ensemble members (default: 20)\n"
    "  -letkf_obs_freq <int>        Observation frequency in coupling intervals (default: 5)\n"
    "  -letkf_obs_stride <int>      Observe every Nth cell (default: 2)\n"
    "  -letkf_obs_error <real>      Observation error std dev (default: 0.01)\n"
    "  -letkf_seed <int>            Random seed (default: 12345)\n"
    "  -letkf_inflation <real>      Ensemble inflation factor (default: 1.0)\n"
    "  -letkf_obs_per_vertex <int>  Local observations per vertex for LETKF (default: 7)\n"
    "  -letkf_steps <int>           Number of coupling intervals to run (default: 0 = use config)\n"
    "  -petscda_type <string>       DA type: etkf or letkf (default: letkf)\n"
    "  -petscda_view                View PetscDA configuration\n\n";

/* Default parameter values */
#define DEFAULT_ENSEMBLE_SIZE    20
#define DEFAULT_OBS_FREQ         5
#define DEFAULT_OBS_STRIDE       2
#define DEFAULT_OBS_ERROR_STD    0.01
#define DEFAULT_SEED             12345
#define DEFAULT_INFLATION        1.0
#define DEFAULT_OBS_PER_VERTEX   7
#define NDOF                     3  /* h, hu, hv per cell */

/* Context for the forecast callback */
typedef struct {
  RDy       rdy;
  PetscReal saved_time;
  PetscInt  saved_step;
} RDyForecastCtx;

/*
  RDyForecastStep - Advance one RDycore coupling interval

  This is the callback passed to PetscDAEnsembleForecast(). It copies the
  input state into RDycore's solution vector, advances one coupling interval
  using TSSolve, and copies the result to the output vector.

  Input Parameters:
+ input - state vector before forecast
- ctx   - RDyForecastCtx containing the RDy instance

  Output Parameter:
. output - state vector after one coupling interval
*/
static PetscErrorCode RDyForecastStep(Vec input, Vec output, PetscCtx ctx)
{
  RDyForecastCtx *fc = (RDyForecastCtx *)ctx;
  RDy             rdy = fc->rdy;

  PetscFunctionBeginUser;

  /* Copy input state into RDycore's solution vector */
  PetscCall(VecCopy(input, rdy->u_global));

  /* Reset TS to the saved time so each ensemble member starts from the same time */
  PetscCall(TSSetTime(rdy->ts, fc->saved_time));
  PetscCall(TSSetStepNumber(rdy->ts, fc->saved_step));

  /* Advance one coupling interval */
  PetscReal interval;
  PetscCall(RDyGetCouplingInterval(rdy, RDY_TIME_SECONDS, &interval));
  PetscCall(TSSetMaxTime(rdy->ts, fc->saved_time + interval));
  PetscCall(TSSetExactFinalTime(rdy->ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTimeStep(rdy->ts, rdy->dt));
  PetscCall(TSSetSolution(rdy->ts, rdy->u_global));
  PetscCall(TSSolve(rdy->ts, rdy->u_global));

  /* Copy result to output */
  if (input != output) {
    PetscCall(VecCopy(rdy->u_global, output));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  CreateSubsampledObservationMatrix - Create observation matrix H that observes
  height (DOF 0) at every obs_stride-th cell in natural ordering.

  Input Parameters:
+ comm       - MPI communicator
. ncells     - total number of cells
. obs_stride - observe every obs_stride-th cell
- ndof       - degrees of freedom per cell

  Output Parameters:
+ H    - observation matrix (nobs x state_size)
- nobs - number of observations
*/
static PetscErrorCode CreateSubsampledObservationMatrix(MPI_Comm comm, PetscInt ncells, PetscInt obs_stride, PetscInt ndof, Mat *H, PetscInt *nobs)
{
  PetscInt state_size = ncells * ndof;
  PetscInt n_obs      = 0;

  PetscFunctionBeginUser;

  /* Count observations */
  for (PetscInt i = 0; i < ncells; i++) {
    if (i % obs_stride == 0) n_obs++;
  }

  /* Create sparse matrix */
  PetscCall(MatCreate(comm, H));
  PetscCall(MatSetSizes(*H, PETSC_DECIDE, PETSC_DECIDE, n_obs, state_size));
  PetscCall(MatSetType(*H, MATAIJ));
  PetscCall(MatSetFromOptions(*H));
  PetscCall(MatSeqAIJSetPreallocation(*H, 1, NULL));
  PetscCall(MatMPIAIJSetPreallocation(*H, 1, NULL, 1, NULL));

  /* Fill: observe height (DOF 0) at subsampled cells */
  PetscInt obs_idx = 0;
  for (PetscInt i = 0; i < ncells; i++) {
    if (i % obs_stride == 0) {
      PetscInt col = i * ndof;  /* DOF 0 = height */
      PetscCall(MatSetValue(*H, obs_idx, col, 1.0, INSERT_VALUES));
      obs_idx++;
    }
  }

  PetscCall(MatAssemblyBegin(*H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*H, MAT_FINAL_ASSEMBLY));

  *nobs = n_obs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  CreateSimpleLocalizationMatrix - Create a localization matrix Q for LETKF.

  For this initial implementation, Q is a (ncells x nobs) matrix where each
  row (grid point) has n_obs_vertex non-zero entries corresponding to the
  nearest observations. For simplicity, we use a banded structure based on
  cell index proximity.

  Input Parameters:
+ comm           - MPI communicator
. ncells         - total number of cells
. nobs           - number of observations
. obs_stride     - observation stride
- n_obs_vertex   - number of local observations per vertex

  Output Parameter:
. Q - localization matrix (ncells x nobs)
*/
static PetscErrorCode CreateSimpleLocalizationMatrix(MPI_Comm comm, PetscInt ncells, PetscInt nobs, PetscInt obs_stride, PetscInt n_obs_vertex, Mat *Q)
{
  PetscFunctionBeginUser;

  PetscCall(MatCreate(comm, Q));
  PetscCall(MatSetSizes(*Q, PETSC_DECIDE, PETSC_DECIDE, ncells, nobs));
  PetscCall(MatSetType(*Q, MATAIJ));
  PetscCall(MatSetFromOptions(*Q));
  PetscCall(MatSeqAIJSetPreallocation(*Q, n_obs_vertex, NULL));
  PetscCall(MatMPIAIJSetPreallocation(*Q, n_obs_vertex, NULL, n_obs_vertex, NULL));

  /* For each grid point, find the nearest n_obs_vertex observations */
  for (PetscInt i = 0; i < ncells; i++) {
    /* Find the observation index closest to this cell */
    PetscInt center_obs = i / obs_stride;
    if (center_obs >= nobs) center_obs = nobs - 1;

    /* Select n_obs_vertex observations centered around center_obs */
    PetscInt half = n_obs_vertex / 2;
    PetscInt start_obs = center_obs - half;
    if (start_obs < 0) start_obs = 0;
    if (start_obs + n_obs_vertex > nobs) start_obs = nobs - n_obs_vertex;
    if (start_obs < 0) start_obs = 0;

    PetscInt actual_count = PetscMin(n_obs_vertex, nobs);
    for (PetscInt k = 0; k < actual_count; k++) {
      PetscInt obs_idx = start_obs + k;
      if (obs_idx >= 0 && obs_idx < nobs) {
        /* Weight = 1.0 (uniform localization for simplicity) */
        PetscCall(MatSetValue(*Q, i, obs_idx, 1.0, INSERT_VALUES));
      }
    }
  }

  PetscCall(MatAssemblyBegin(*Q, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*Q, MAT_FINAL_ASSEMBLY));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ComputeRMSE - Compute root mean square error between two vectors
*/
static PetscErrorCode ComputeRMSE(Vec a, Vec b, Vec work, PetscInt n, PetscReal *rmse)
{
  PetscFunctionBeginUser;
  PetscCall(VecCopy(a, work));
  PetscCall(VecAXPY(work, -1.0, b));
  PetscReal norm;
  PetscCall(VecNorm(work, NORM_2, &norm));
  *rmse = norm / PetscSqrtReal((PetscReal)n);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* VecSetRandomGaussian is provided by petsc_gem's petscvec.h */

int main(int argc, char *argv[]) {
  /* Print usage if no arguments */
  if (argc < 2) {
    fprintf(stderr, "rdycore_letkf: usage:\n");
    fprintf(stderr, "rdycore_letkf <input.yaml> [options]\n\n");
    exit(0);
  }

  /* Initialize */
  PetscCall(RDyInit(argc, argv, help_str));

  if (strcmp(argv[1], "-help")) {
    MPI_Comm comm = PETSC_COMM_WORLD;
    PetscMPIInt comm_size;
    PetscCallMPI(MPI_Comm_size(comm, &comm_size));

    /* Parse LETKF options */
    PetscInt  ensemble_size    = DEFAULT_ENSEMBLE_SIZE;
    PetscInt  obs_freq         = DEFAULT_OBS_FREQ;
    PetscInt  obs_stride       = DEFAULT_OBS_STRIDE;
    PetscReal obs_error_std    = DEFAULT_OBS_ERROR_STD;
    PetscInt  random_seed      = DEFAULT_SEED;
    PetscReal inflation        = DEFAULT_INFLATION;
    PetscInt  n_obs_per_vertex = DEFAULT_OBS_PER_VERTEX;
    PetscInt  letkf_steps      = 0;

    PetscCall(PetscOptionsGetInt(NULL, NULL, "-letkf_ensemble_size", &ensemble_size, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-letkf_obs_freq", &obs_freq, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-letkf_obs_stride", &obs_stride, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-letkf_obs_error", &obs_error_std, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-letkf_seed", &random_seed, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-letkf_inflation", &inflation, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-letkf_obs_per_vertex", &n_obs_per_vertex, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-letkf_steps", &letkf_steps, NULL));

    /* Create and set up RDycore */
    RDy rdy;
    PetscCall(RDyCreate(comm, argv[1], &rdy));
    PetscCall(RDySetup(rdy));

    /* Get mesh dimensions */
    PetscInt ncells_local, ncells_global;
    PetscCall(RDyGetNumLocalCells(rdy, &ncells_local));
    PetscCall(RDyGetNumGlobalCells(rdy, &ncells_global));

    PetscInt state_size = ncells_global * NDOF;

    /* Determine number of steps */
    PetscInt total_steps;
    if (letkf_steps > 0) {
      total_steps = letkf_steps;
    } else {
      /* Compute from config: final_time / coupling_interval */
      PetscReal final_time, coupling_interval;
      RDyTimeUnit time_unit;
      PetscCall(RDyGetTimeUnit(rdy, &time_unit));
      final_time        = rdy->config.time.final_time;
      PetscCall(RDyGetCouplingInterval(rdy, time_unit, &coupling_interval));
      total_steps = (PetscInt)(final_time / coupling_interval + 0.5);
      if (total_steps < 1) total_steps = 1;
    }

    /* Create observation matrix H */
    Mat      H;
    PetscInt nobs;
    PetscCall(CreateSubsampledObservationMatrix(comm, ncells_global, obs_stride, NDOF, &H, &nobs));

    /* Ensure n_obs_per_vertex doesn't exceed nobs */
    if (n_obs_per_vertex > nobs) n_obs_per_vertex = nobs;

    /* Create random number generator */
    PetscRandom rng;
    PetscCall(PetscRandomCreate(comm, &rng));
    PetscCall(PetscRandomSetType(rng, PETSCRAND48));
    PetscCall(PetscRandomSetSeed(rng, random_seed));
    PetscCall(PetscRandomSeed(rng));
    PetscCall(PetscRandomSetInterval(rng, 0.0, 1.0));

    /* Create PetscDA */
    PetscDA da;
    PetscCall(PetscDACreate(comm, &da));
    PetscCall(PetscDASetFromOptions(da));
    PetscCall(PetscDASetSizes(da, state_size, nobs));
    PetscCall(PetscDASetNDOF(da, NDOF));
    PetscCall(PetscDAEnsembleSetSize(da, ensemble_size));

    if (inflation != 1.0) {
      PetscCall(PetscDAEnsembleSetInflation(da, inflation));
    }

    /* Set observation error variance */
    Vec obs_error_var;
    PetscCall(MatCreateVecs(H, NULL, &obs_error_var));
    PetscCall(VecSet(obs_error_var, obs_error_std * obs_error_std));
    PetscCall(PetscDASetObsErrorVariance(da, obs_error_var));

    /* Create localization matrix Q and set up LETKF */
    {
      Mat Q;
      PetscCall(CreateSimpleLocalizationMatrix(comm, ncells_global, nobs, obs_stride, n_obs_per_vertex, &Q));
      PetscCall(PetscDALETKFSetLocalization(da, Q, H));
      PetscCall(PetscDALETKFSetObsPerVertex(da, n_obs_per_vertex));
      PetscCall(MatDestroy(&Q));
    }

    PetscCall(PetscDASetUp(da));

    /* Initialize ensemble from truth state + perturbations */
    PetscCall(PetscDAEnsembleInitialize(da, rdy->u_global, obs_error_std, rng));

    /* View DA configuration */
    PetscCall(PetscDAViewFromOptions(da, NULL, "-petscda_view"));

    /* Create truth state (copy of initial condition) */
    Vec truth_state;
    PetscCall(VecDuplicate(rdy->u_global, &truth_state));
    PetscCall(VecCopy(rdy->u_global, truth_state));

    /* Create work vectors */
    Vec x_mean, x_forecast, rmse_work;
    PetscCall(VecDuplicate(rdy->u_global, &x_mean));
    PetscCall(VecDuplicate(rdy->u_global, &x_forecast));
    PetscCall(VecDuplicate(rdy->u_global, &rmse_work));

    /* Create observation vectors */
    Vec observation, obs_noise;
    PetscCall(MatCreateVecs(H, NULL, &observation));
    PetscCall(VecDuplicate(observation, &obs_noise));

    /* Set up forecast context */
    RDyForecastCtx forecast_ctx;
    forecast_ctx.rdy = rdy;

    /* Print configuration */
    PetscCall(PetscPrintf(comm, "RDycore LETKF Twin Experiment\n"));
    PetscCall(PetscPrintf(comm, "=============================\n"));
    PetscCall(PetscPrintf(comm, "  Config file           : %s\n", argv[1]));
    PetscCall(PetscPrintf(comm, "  Global cells          : %" PetscInt_FMT "\n", ncells_global));
    PetscCall(PetscPrintf(comm, "  State dimension       : %" PetscInt_FMT " (%" PetscInt_FMT " cells x %d DOF)\n", state_size, ncells_global, NDOF));
    PetscCall(PetscPrintf(comm, "  Observation dimension : %" PetscInt_FMT "\n", nobs));
    PetscCall(PetscPrintf(comm, "  Observation stride    : %" PetscInt_FMT "\n", obs_stride));
    PetscCall(PetscPrintf(comm, "  Ensemble size         : %" PetscInt_FMT "\n", ensemble_size));
    PetscCall(PetscPrintf(comm, "  Total steps           : %" PetscInt_FMT "\n", total_steps));
    PetscCall(PetscPrintf(comm, "  Observation frequency : %" PetscInt_FMT "\n", obs_freq));
    PetscCall(PetscPrintf(comm, "  Observation noise std : %.4f\n", (double)obs_error_std));
    PetscCall(PetscPrintf(comm, "  Inflation factor      : %.4f\n", (double)inflation));
    PetscCall(PetscPrintf(comm, "  Obs per vertex        : %" PetscInt_FMT "\n", n_obs_per_vertex));
    PetscCall(PetscPrintf(comm, "  Random seed           : %" PetscInt_FMT "\n", random_seed));
    PetscCall(PetscPrintf(comm, "  MPI processes         : %d\n\n", (int)comm_size));

    /* Print initial RMSE */
    PetscCall(PetscDAEnsembleComputeMean(da, x_mean));
    PetscReal rmse_initial;
    PetscCall(ComputeRMSE(x_mean, truth_state, rmse_work, state_size, &rmse_initial));
    PetscCall(PetscPrintf(comm, "Step %4d  RMSE_forecast %.6f  RMSE_analysis %.6f [initial]\n", 0, (double)rmse_initial, (double)rmse_initial));

    /* Main DA loop */
    PetscReal sum_rmse_forecast = 0.0, sum_rmse_analysis = 0.0;
    PetscInt  obs_count = 0;

    for (PetscInt step = 1; step <= total_steps; step++) {
      PetscReal rmse_forecast, rmse_analysis;

      /* Save TS state before forecast */
      PetscCall(TSGetTime(rdy->ts, &forecast_ctx.saved_time));
      PetscCall(TSGetStepNumber(rdy->ts, &forecast_ctx.saved_step));

      /* Forecast: advance all ensemble members */
      PetscCall(PetscDAEnsembleForecast(da, RDyForecastStep, &forecast_ctx));

      /* Advance truth state */
      PetscCall(RDyForecastStep(truth_state, truth_state, &forecast_ctx));

      /* Update the saved time to the new time (after forecast) */
      PetscCall(TSGetTime(rdy->ts, &forecast_ctx.saved_time));
      PetscCall(TSGetStepNumber(rdy->ts, &forecast_ctx.saved_step));

      /* Compute forecast RMSE */
      PetscCall(PetscDAEnsembleComputeMean(da, x_mean));
      PetscCall(VecCopy(x_mean, x_forecast));
      PetscCall(ComputeRMSE(x_forecast, truth_state, rmse_work, state_size, &rmse_forecast));
      rmse_analysis = rmse_forecast;

      /* Analysis step: assimilate observations when available */
      if (step % obs_freq == 0) {
        Vec truth_obs, temp_truth;
        PetscCall(MatCreateVecs(H, NULL, &truth_obs));
        PetscCall(MatCreateVecs(H, &temp_truth, NULL));

        /* Generate observations from truth: obs = H * truth + noise */
        PetscCall(VecCopy(truth_state, temp_truth));
        PetscCall(MatMult(H, temp_truth, truth_obs));

        /* Add observation noise */
        PetscCall(VecSetRandomGaussian(obs_noise, rng, 0.0, obs_error_std));
        PetscCall(VecWAXPY(observation, 1.0, obs_noise, truth_obs));

        /* Perform LETKF analysis */
        PetscCall(PetscDAEnsembleAnalysis(da, observation, H));

        /* Clean up */
        PetscCall(VecDestroy(&temp_truth));
        PetscCall(VecDestroy(&truth_obs));

        /* Compute analysis RMSE */
        PetscCall(PetscDAEnsembleComputeMean(da, x_mean));
        PetscCall(ComputeRMSE(x_mean, truth_state, rmse_work, state_size, &rmse_analysis));
        obs_count++;
      }

      /* Accumulate statistics */
      sum_rmse_forecast += rmse_forecast;
      sum_rmse_analysis += rmse_analysis;

      /* Progress reporting */
      PetscCall(PetscPrintf(comm, "Step %4" PetscInt_FMT "  RMSE_forecast %.6f  RMSE_analysis %.6f%s\n", step, (double)rmse_forecast, (double)rmse_analysis,
                            (step % obs_freq == 0) ? " [obs]" : ""));
    }

    /* Report final statistics */
    if (total_steps > 0) {
      PetscReal avg_rmse_forecast = sum_rmse_forecast / total_steps;
      PetscReal avg_rmse_analysis = sum_rmse_analysis / total_steps;
      PetscCall(PetscPrintf(comm, "\nStatistics (%" PetscInt_FMT " steps):\n", total_steps));
      PetscCall(PetscPrintf(comm, "==================================================\n"));
      PetscCall(PetscPrintf(comm, "  Mean RMSE (forecast) : %.6f\n", (double)avg_rmse_forecast));
      PetscCall(PetscPrintf(comm, "  Mean RMSE (analysis) : %.6f\n", (double)avg_rmse_analysis));
      PetscCall(PetscPrintf(comm, "  Observations used    : %" PetscInt_FMT "\n\n", obs_count));
    }

    /* Cleanup */
    PetscCall(MatDestroy(&H));
    PetscCall(VecDestroy(&obs_error_var));
    PetscCall(VecDestroy(&observation));
    PetscCall(VecDestroy(&obs_noise));
    PetscCall(VecDestroy(&x_forecast));
    PetscCall(VecDestroy(&x_mean));
    PetscCall(VecDestroy(&rmse_work));
    PetscCall(VecDestroy(&truth_state));
    PetscCall(PetscRandomDestroy(&rng));
    PetscCall(PetscDADestroy(&da));
    PetscCall(RDyDestroy(&rdy));
  }

  PetscCall(RDyFinalize());
  return 0;
}
