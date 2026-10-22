#include <petscsys.h>
#include <rdycore.h>
#include <time.h>

static const char *help_str =
    "rdycore_amr - a standalone RDycore driver for AMR problems\n"
    "usage: rdycore_amr [options] <filename.yaml>\n"
    "\n"
    "  -refine_criterion <box|jump>  what to refine on: a fixed box (default), or\n"
    "                                the water-surface jump of the solution\n"
    "  -refine_at_start              adapt to the initial condition before stepping\n"
    "  -refine_allow_coarsening      rebuild the mesh from the base mesh at every\n"
    "                                adapt, so cells the criterion no longer marks\n"
    "                                come back coarse\n"
    "  -refine_check_volume          report sum(h * area) across every adapt\n"
    "  -refine_data_dir <dir>        refine from marker files instead (with\n"
    "                                -refine_data_start_date Y,M,D,h,m)\n"
    "\n"
    "  the jump criterion is tuned with -rdy_amr_vec_tagger_type <relative|cdf|absolute>\n"
    "  and -rdy_amr_vec_tagger_box <lo,hi>, -rdy_amr_buffer_layers <n> and\n"
    "  -rdy_amr_max_levels <n>\n"
    "\n"
    "  the mesh must be simplicial and the adapt needs\n"
    "  -dm_adaptor cellrefiner -dm_plex_transform_type refine_sbr\n"
    "  -dm_plex_transform_label_match_strata\n";

static void usage(const char *exe_name) {
  fprintf(stderr, "%s: usage:\n", exe_name);
  fprintf(stderr, "%s <input.yaml>\n\n", exe_name);
}

typedef struct {
  char dir[PETSC_MAX_PATH_LEN];
  char file[PETSC_MAX_PATH_LEN];

  PetscBool is_defined;          // whether to refine or not
  struct tm date, current_date;  // date for refinement
} RefinementDataset;

static PetscErrorCode DetermineDatasetFilename(struct tm *current_date, char *dir, char *file) {
  PetscFunctionBegin;

  mktime(current_date);
  snprintf(file, PETSC_MAX_PATH_LEN - 1, "%s/%4d-%02d-%02d:%02d-%02d.%s.bin", dir, current_date->tm_year + 1900, current_date->tm_mon + 1,
           current_date->tm_mday, current_date->tm_hour, current_date->tm_min, PETSC_ID_TYPE);

  PetscFunctionReturn(PETSC_SUCCESS);
}

#define NUM_DATE_VALUES 5  // number of parameters expected for date
PetscErrorCode ParseRefinementDataOptions(RefinementDataset *refinement_dataset) {
  PetscFunctionBegin;

  // set default settings
  refinement_dataset->is_defined = PETSC_FALSE;

  // parse command line options
  PetscInt date[NUM_DATE_VALUES];
  PetscInt ndate = NUM_DATE_VALUES;

  PetscBool refine_flag;
  PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-refine_data_start_date", date, &ndate, &refine_flag));

  PetscBool dir_flag;
  PetscCall(PetscOptionsGetString(NULL, NULL, "-refine_data_dir", refinement_dataset->dir, sizeof(refinement_dataset->dir), &dir_flag));

  if (refine_flag) {
    PetscCheck(dir_flag == PETSC_TRUE, PETSC_COMM_WORLD, PETSC_ERR_USER,
               "Need to specify path to spatially raster rainfall via -refine_data_dir <dir>");

    refinement_dataset->is_defined = PETSC_TRUE;
    refinement_dataset->date       = (struct tm){
              .tm_year  = date[0] - 1900,
              .tm_mon   = date[1] - 1,
              .tm_mday  = date[2],
              .tm_hour  = date[3],
              .tm_min   = date[4],
              .tm_isdst = -1,
    };

    refinement_dataset->current_date = (struct tm){
        .tm_year  = date[0] - 1900,
        .tm_mon   = date[1] - 1,
        .tm_mday  = date[2],
        .tm_hour  = date[3],
        .tm_min   = date[4],
        .tm_isdst = -1,
    };
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MarkOwnedCellsForRefinementBasedOnDataset(RDy rdy, Vec global) {
  PetscFunctionBegin;

  PetscInt     ncells_local;
  PetscBool   *refine_cell;
  PetscReal   *area_local;
  PetscScalar *global_ptr;

  // determine the number of owned cells
  PetscCall(RDyGetNumOwnedCells(rdy, &ncells_local));

  // allocate memory
  PetscCalloc1(ncells_local, &refine_cell);
  PetscCalloc1(ncells_local, &area_local);

  PetscCall(RDyGetOwnedCellAreas(rdy, ncells_local, area_local));

  PetscReal area_threshold = 1.0 / 4.0 / 2.0;  // m^2

  PetscCall(VecGetArray(global, &global_ptr));
  for (PetscInt icell = 0; icell < ncells_local; icell++) {
    if (global_ptr[icell] > 0.0 && area_local[icell] > area_threshold) {
      refine_cell[icell] = PETSC_TRUE;
    } else {
      refine_cell[icell] = PETSC_FALSE;
    }
  }
  PetscCall(VecRestoreArray(global, &global_ptr));

  PetscCall(RDyMarkOwnedCellsForAMR(rdy, ncells_local, refine_cell));

  // free up memory
  PetscFree(refine_cell);
  PetscFree(area_local);

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Reports the water volume the mesh holds, sum(h * area) over owned
///        cells. Refinement and the interpolation that follows it should not
///        change this: comparing the value across an adapt is the conservation
///        check for the AMR path.
/// @param [in] rdy    the dycore
/// @param [out] total the global water volume (m^3), valid on every rank
static PetscErrorCode ComputeWaterVolume(RDy rdy, PetscReal *total) {
  PetscFunctionBegin;

  PetscInt ncells_local;
  PetscCall(RDyGetNumOwnedCells(rdy, &ncells_local));

  PetscReal *h, *area;
  PetscCalloc1(ncells_local, &h);
  PetscCalloc1(ncells_local, &area);
  PetscCall(RDyGetOwnedCellHeights(rdy, ncells_local, h));
  PetscCall(RDyGetOwnedCellAreas(rdy, ncells_local, area));

  PetscReal local = 0.0;
  for (PetscInt icell = 0; icell < ncells_local; icell++) local += h[icell] * area[icell];
  PetscCallMPI(MPI_Allreduce(&local, total, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));

  PetscFree(h);
  PetscFree(area);

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MarkOwnedCellsForRefinement(RDy rdy) {
  PetscFunctionBegin;

  PetscInt   ncells_local;
  PetscReal *xc_local, *yc_local;
  PetscBool *refine_cell;

  // determine the number of owned cells
  PetscCall(RDyGetNumOwnedCells(rdy, &ncells_local));

  // allocate memory
  PetscCalloc1(ncells_local, &xc_local);
  PetscCalloc1(ncells_local, &yc_local);
  PetscCalloc1(ncells_local, &refine_cell);

  // get the x and y coordinates of the local cells
  PetscCall(RDyGetOwnedCellXCentroids(rdy, ncells_local, xc_local));
  PetscCall(RDyGetOwnedCellYCentroids(rdy, ncells_local, yc_local));

  for (PetscInt icell = 0; icell < ncells_local; icell++) {
    // check if the cell is inside the region of interest
    if (xc_local[icell] > 0.0 && xc_local[icell] < 1.0 && yc_local[icell] > 3.0 && yc_local[icell] < 4.0) {
      // mark the cell for refinement
      refine_cell[icell] = PETSC_TRUE;
    } else {
      // do not mark the cell for refinement
      refine_cell[icell] = PETSC_FALSE;
    }
  }

  PetscCall(RDyMarkOwnedCellsForAMR(rdy, ncells_local, refine_cell));

  // free up memory
  PetscFree(xc_local);
  PetscFree(yc_local);
  PetscFree(refine_cell);

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[]) {
  // print usage info if no arguments given
  if (argc < 2) {
    usage(argv[0]);
    exit(-1);
  }

  // initialize subsystems
  PetscCall(RDyInit(argc, argv, help_str));

  PetscMPIInt myrank, commsize;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &myrank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &commsize));

  if (strcmp(argv[1], "-help")) {  // if given a config file

    // create rdycore and set it up with the given file
    MPI_Comm comm = PETSC_COMM_WORLD;
    RDy      rdy;

    // set things up
    PetscCall(RDyCreate(comm, argv[1], &rdy));
    PetscCall(RDySetup(rdy));

    // enable refinement
    PetscBool by_jump_early = PETSC_FALSE;
    {
      char criterion[32] = "box";
      PetscCall(PetscOptionsGetString(NULL, NULL, "-refine_criterion", criterion, sizeof(criterion), NULL));
      PetscCall(PetscStrcmp(criterion, "jump", &by_jump_early));
    }
    if (by_jump_early) {
      PetscCall(RDyEnableAMRBySurfaceJump(rdy));  // the DMAdaptor marks the cells
    } else {
      PetscCall(RDyEnableAMR(rdy));
    }

    RefinementDataset refinement_dataset;
    PetscCall(ParseRefinementDataOptions(&refinement_dataset));

    // -refine_criterion speed marks cells by the local flow speed instead of
    // the static box of MarkOwnedCellsForRefinement, so the mesh follows the
    // solution. -refine_check_volume reports sum(h * area) across each adapt.
    char      criterion[32] = "box";
    PetscBool check_volume = PETSC_FALSE, refine_at_start = PETSC_FALSE;
    PetscCall(PetscOptionsGetString(NULL, NULL, "-refine_criterion", criterion, sizeof(criterion), NULL));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-refine_check_volume", &check_volume, NULL));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-refine_at_start", &refine_at_start, NULL));
    PetscBool by_jump;
    PetscCall(PetscStrcmp(criterion, "jump", &by_jump));
    if (!by_jump) {
      PetscBool by_box;
      PetscCall(PetscStrcmp(criterion, "box", &by_box));
      PetscCheck(by_box, comm, PETSC_ERR_USER, "-refine_criterion must be 'box' or 'jump', got '%s'", criterion);
    }

    RDyTimeUnit time_unit;
    PetscReal   time, prev_time;
    PetscCall(RDyGetTimeUnit(rdy, &time_unit));
    PetscCall(RDyGetTime(rdy, time_unit, &prev_time));

    if (refinement_dataset.is_defined) {
      // determine the filename
      PetscCall(DetermineDatasetFilename(&refinement_dataset.current_date, refinement_dataset.dir, refinement_dataset.file));
      if (!myrank) printf("refinement filename: %s\n", refinement_dataset.file);
      Vec global;
      PetscCall(RDyReadAMRScalarGlobalVecLevel0FromBinary(rdy, refinement_dataset.file, &global));

      PetscCall(MarkOwnedCellsForRefinementBasedOnDataset(rdy, global));
      PetscCall(VecDestroy(&global));
      PetscCall(RDyPerformAMR(rdy));
    }

    if (refine_at_start && by_jump) {
      PetscCall(RDyPerformAMR(rdy));
      PetscInt ncells;
      PetscCall(RDyGetNumGlobalCells(rdy, &ncells));
      if (!myrank) printf("AMR: %" PetscInt_FMT " cells after adapting to the initial condition\n", ncells);
    }

    PetscCall(RDyAdvance(rdy));

    PetscCall(RDyGetTime(rdy, time_unit, &time));

    while (!RDyFinished(rdy)) {
      if (refinement_dataset.is_defined) {
        // increase the date
        struct tm *current_date = &refinement_dataset.current_date;
        current_date->tm_min++;
        mktime(current_date);

        PetscCall(DetermineDatasetFilename(&refinement_dataset.current_date, refinement_dataset.dir, refinement_dataset.file));
        if (!myrank) printf("refinement filename: %s\n", refinement_dataset.file);
        Vec global_base, global_current;
        PetscCall(RDyReadAMRScalarGlobalVecLevel0FromBinary(rdy, refinement_dataset.file, &global_base));
        PetscCall(RDyMapAMRScalarGlobalVecLevel0ToCurrentLevel(rdy, global_base, &global_current));
        PetscCall(MarkOwnedCellsForRefinementBasedOnDataset(rdy, global_current));

        PetscCall(VecDestroy(&global_base));
        PetscCall(VecDestroy(&global_current));
      } else if (by_jump) {
        // nothing to do: the DMAdaptor computes the indicator inside the adapt
      } else {
        PetscCall(MarkOwnedCellsForRefinement(rdy));
      }

      PetscReal volume_before = 0.0, volume_after = 0.0;
      if (check_volume) PetscCall(ComputeWaterVolume(rdy, &volume_before));

      PetscCall(RDyPerformAMR(rdy));

      if (by_jump) {
        PetscInt ncells;
        PetscCall(RDyGetNumGlobalCells(rdy, &ncells));
        if (!myrank) printf("AMR: %" PetscInt_FMT " cells after adapt\n", ncells);
      }

      if (check_volume) {
        PetscCall(ComputeWaterVolume(rdy, &volume_after));
        PetscReal rel = (volume_before != 0.0) ? PetscAbsReal(volume_after - volume_before) / volume_before : 0.0;
        if (!myrank) printf("AMR: water volume %.15g -> %.15g m^3, relative change %.3e\n", (double)volume_before, (double)volume_after, (double)rel);
      }

      PetscCall(RDyAdvance(rdy));
    }

    PetscCall(RDyDestroy(&rdy));
  }

  PetscCall(RDyFinalize());
  return 0;
}
