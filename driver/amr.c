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
    "                                (-refine_criterion jump only; the box does not\n"
    "                                move before the first step)\n"
    "  -refine_allow_coarsening      rebuild the mesh from the base mesh at every\n"
    "                                adapt, so cells the criterion no longer marks\n"
    "                                come back coarse\n"
    "  -refine_check_volume          check sum(h * area) across every adapt, and\n"
    "                                fail if it moves by more than\n"
    "                                -refine_check_volume_tol (default 1e-11)\n"
    "                                relative\n"
    "  -refine_check_volume_tol <t>  that tolerance\n"
    "  -refine_report_balance        report max/mean owned cells over the ranks\n"
    "                                on either side of every adapt\n"
    "  -refine_expect_water_volume <v>  fail unless the run ends holding v m^3,\n"
    "                                within -refine_expect_water_volume_tol\n"
    "                                (default 1e-9) relative\n"
    "  -rdy_amr_balance 0            do not repartition the base mesh at each\n"
    "                                rebuild (on by default, and only a rebuild,\n"
    "                                so -refine_allow_coarsening, can balance)\n"
    "  -refine_check_max_level       report how deep the mesh has been refined\n"
    "                                after every adapt, and fail more than one\n"
    "                                level past -rdy_amr_max_levels (which caps\n"
    "                                marking; the conformity closure can add one)\n"
    "  -refine_data_dir <dir>        refine from marker files instead (with\n"
    "                                -refine_data_start_date Y,M,D,h,m)\n"
    "\n"
    "  -refine_box <x0,x1,y0,y1>     the box the box criterion marks\n"
    "  -refine_box_speed <v>         translate that box along x at v m/s, so\n"
    "                                that cells behind it coarsen again\n"
    "\n"
    "  -refine_still_water <eta>     start from a lake at rest at elevation eta\n"
    "                                (h = eta - z) and report how far the free\n"
    "                                surface departs from it on either side of\n"
    "                                every adapt\n"
    "  -refine_still_water_tol <t>   fail if that departure exceeds t * eta\n"
    "\n"
    "  the jump criterion is tuned with\n"
    "  -rdy_amr_refine_vec_tagger_type <relative|cdf|absolute>,\n"
    "  -rdy_amr_refine_vec_tagger_box <lo,hi>,\n"
    "  -rdy_amr_coarsen_vec_tagger_type <relative|cdf|absolute>,\n"
    "  -rdy_amr_coarsen_vec_tagger_box <lo,hi>,\n"
    "  -rdy_amr_min_jump <h>, -rdy_amr_buffer_layers <n>,\n"
    "  -rdy_amr_max_levels <n>, and -rdy_amr_error_vec_view\n"
    "\n"
    "  the mesh must be simplicial; the adapt refines it by bisection\n";

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
  PetscInt date[NUM_DATE_VALUES] = {0};
  PetscInt ndate                 = NUM_DATE_VALUES;

  PetscBool refine_flag      = PETSC_FALSE;
  PetscBool dir_flag         = PETSC_FALSE;
  refinement_dataset->dir[0] = '\0';
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "AMR dataset refinement options", "");
  {
    PetscCall(PetscOptionsIntArray("-refine_data_start_date", "Start date Y,M,D,h,m for marker-file refinement", "", date, &ndate, &refine_flag));
    PetscCall(PetscOptionsString("-refine_data_dir", "Directory containing marker files for refinement", "", refinement_dataset->dir,
                                 refinement_dataset->dir, sizeof(refinement_dataset->dir), &dir_flag));
  }
  PetscOptionsEnd();

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
  PetscCall(PetscCalloc1(ncells_local, &refine_cell));
  PetscCall(PetscCalloc1(ncells_local, &area_local));

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
  PetscCall(PetscFree(refine_cell));
  PetscCall(PetscFree(area_local));

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
  PetscCall(PetscCalloc1(ncells_local, &h));
  PetscCall(PetscCalloc1(ncells_local, &area));
  PetscCall(RDyGetOwnedCellHeights(rdy, ncells_local, h));
  PetscCall(RDyGetOwnedCellAreas(rdy, ncells_local, area));

  PetscReal local = 0.0;
  for (PetscInt icell = 0; icell < ncells_local; icell++) local += h[icell] * area[icell];
  PetscCallMPI(MPI_Allreduce(&local, total, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));

  PetscCall(PetscFree(h));
  PetscCall(PetscFree(area));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Reports, and fails on, the deepest any owned cell has been refined
///        below the base cell it descends from. A bisection halves a cell
///        exactly, so a cell `level` below its base cell has 2^-level of its
///        area, whatever the base cell's size -- which is the point: a limit
///        expressed as one area for the whole mesh binds only on the finest
///        base cell, and on a graded mesh every coarser one refines past it.
///
///        The limit allowed here is max_levels + 1, not max_levels.
///        -rdy_amr_max_levels caps the level a cell may be MARKED at;
///        refine_sbr then splits a marked triangle into two children, or into
///        three of areas 1/2, 1/4, 1/4 where conformity needs a second edge
///        split, so a cell marked at max_levels - 1 can come back at
///        max_levels + 1. Measured, not proved: a longer conformity chain
///        through cells that are already at the limit is not ruled out by
///        anything we have read, and a failure here would be worth knowing
///        about rather than tolerating.
static PetscErrorCode CheckRefinementLimit(RDy rdy, PetscInt max_levels) {
  PetscFunctionBegin;

  PetscInt ncells_local;
  PetscCall(RDyGetNumOwnedCells(rdy, &ncells_local));

  PetscReal *area, *base_area;
  PetscCall(PetscCalloc1(ncells_local, &area));
  PetscCall(PetscCalloc1(ncells_local, &base_area));
  PetscCall(RDyGetOwnedCellAreas(rdy, ncells_local, area));
  PetscCall(RDyGetOwnedCellBaseAreas(rdy, ncells_local, base_area));

  PetscInt local = 0, deepest;
  for (PetscInt icell = 0; icell < ncells_local; icell++) {
    // the ratio is a power of two, so rounding the logarithm absorbs roundoff
    local = PetscMax(local, (PetscInt)PetscFloorReal(PetscLog2Real(base_area[icell] / area[icell]) + 0.5));
  }
  PetscCallMPI(MPI_Allreduce(&local, &deepest, 1, MPIU_INT, MPI_MAX, PETSC_COMM_WORLD));

  PetscCall(PetscFree(area));
  PetscCall(PetscFree(base_area));

  PetscMPIInt myrank;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &myrank));
  if (!myrank) {
    printf("AMR: deepest cell is %" PetscInt_FMT " levels below its base cell (marking limit %" PetscInt_FMT ")\n", deepest, max_levels);
  }
  PetscCheck(deepest <= max_levels + 1, PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED,
             "the adapt refined past -rdy_amr_max_levels: a cell is %" PetscInt_FMT
             " levels below its base cell, and the conformity closure accounts for at most %" PetscInt_FMT,
             deepest, max_levels + 1);

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Reports how unevenly the adapted mesh is spread over the ranks:
///        the largest number of owned cells on any rank against the mean.
///
///        Refinement concentrates on the feature, so the rank that owns it
///        ends up doing most of the work while the others wait at the next
///        halo exchange. The ratio is what load balancing has to bring down;
///        1.0 is perfect.
static PetscErrorCode ReportBalance(RDy rdy, const char when[]) {
  PetscFunctionBegin;

  PetscInt ncells_local, maxcells, sumcells;
  PetscCall(RDyGetNumOwnedCells(rdy, &ncells_local));
  PetscCallMPI(MPI_Allreduce(&ncells_local, &maxcells, 1, MPIU_INT, MPI_MAX, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&ncells_local, &sumcells, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));

  PetscMPIInt size, myrank;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &myrank));
  PetscReal mean = (PetscReal)sumcells / size;
  if (!myrank) {
    printf("AMR: owned cells %s the adapt: max %" PetscInt_FMT ", mean %.1f, max/mean %.3f\n", when, maxcells, (double)mean,
           (double)(maxcells / mean));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Marks the cells whose centroid lies in `box` = {x0, x1, y0, y1},
///        shifted along x by `shift`.
PetscErrorCode MarkOwnedCellsForRefinement(RDy rdy, const PetscReal box[4], PetscReal shift) {
  PetscFunctionBegin;

  PetscInt   ncells_local;
  PetscReal *xc_local, *yc_local;
  PetscBool *refine_cell;

  // determine the number of owned cells
  PetscCall(RDyGetNumOwnedCells(rdy, &ncells_local));

  // allocate memory
  PetscCall(PetscCalloc1(ncells_local, &xc_local));
  PetscCall(PetscCalloc1(ncells_local, &yc_local));
  PetscCall(PetscCalloc1(ncells_local, &refine_cell));

  // get the x and y coordinates of the local cells
  PetscCall(RDyGetOwnedCellXCentroids(rdy, ncells_local, xc_local));
  PetscCall(RDyGetOwnedCellYCentroids(rdy, ncells_local, yc_local));

  for (PetscInt icell = 0; icell < ncells_local; icell++) {
    PetscReal x = xc_local[icell] - shift, y = yc_local[icell];
    refine_cell[icell] = (PetscBool)(x > box[0] && x < box[1] && y > box[2] && y < box[3]);
  }

  PetscCall(RDyMarkOwnedCellsForAMR(rdy, ncells_local, refine_cell));

  // free up memory
  PetscCall(PetscFree(xc_local));
  PetscCall(PetscFree(yc_local));
  PetscCall(PetscFree(refine_cell));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Sets a lake at rest: still water whose free surface is everywhere at
///        `eta`, so the depth h = eta - z varies from cell to cell wherever the
///        bed does.
static PetscErrorCode SetStillWater(RDy rdy, PetscReal eta) {
  PetscFunctionBegin;

  PetscInt ncells_local;
  PetscCall(RDyGetNumOwnedCells(rdy, &ncells_local));

  PetscReal *z;
  PetscCall(PetscCalloc1(ncells_local, &z));
  PetscCall(RDyGetOwnedCellZCentroids(rdy, ncells_local, z));

  Vec ic;
  PetscCall(RDyCreatePrognosticVec(rdy, &ic));
  PetscInt ndof;
  PetscCall(VecGetBlockSize(ic, &ndof));

  PetscScalar *u;
  PetscCall(VecZeroEntries(ic));
  PetscCall(VecGetArray(ic, &u));
  for (PetscInt icell = 0; icell < ncells_local; icell++) u[ndof * icell] = PetscMax(eta - z[icell], 0.0);
  PetscCall(VecRestoreArray(ic, &u));

  PetscCall(RDySetInitialConditions(rdy, ic));
  PetscCall(VecDestroy(&ic));
  PetscCall(PetscFree(z));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief How far the free surface has departed from a lake at rest at `eta`:
///        max|h + z - eta| over the wet owned cells, reduced over all ranks.
///        Dry cells are excluded -- there the surface is the bed, not eta.
static PetscErrorCode StillWaterDeparture(RDy rdy, PetscReal eta, PetscReal *departure) {
  PetscFunctionBegin;

  PetscInt ncells_local;
  PetscCall(RDyGetNumOwnedCells(rdy, &ncells_local));

  PetscReal *h, *z;
  PetscCall(PetscCalloc1(ncells_local, &h));
  PetscCall(PetscCalloc1(ncells_local, &z));
  PetscCall(RDyGetOwnedCellHeights(rdy, ncells_local, h));
  PetscCall(RDyGetOwnedCellZCentroids(rdy, ncells_local, z));

  PetscReal local = 0.0;
  for (PetscInt icell = 0; icell < ncells_local; icell++) {
    if (h[icell] > 0.0) local = PetscMax(local, PetscAbsReal(h[icell] + z[icell] - eta));
  }
  PetscCallMPI(MPI_Allreduce(&local, departure, 1, MPIU_REAL, MPI_MAX, PETSC_COMM_WORLD));

  PetscCall(PetscFree(h));
  PetscCall(PetscFree(z));

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

    // enable refinement. -refine_criterion jump marks cells by the water-surface
    // jump of the solution instead of the box of MarkOwnedCellsForRefinement, so
    // the mesh follows the solution; it has to be known before RDyEnableAMR.
    //
    // sum(h * area) is what the transfer promises to preserve: the area-weighted
    // average reproduces it exactly and only the summation order changes, so the
    // departure is machine precision (measured: at most 1.4e-14 relative over
    // every case in this suite). Reporting it is not a check -- the number has to
    // be held to something, or a transfer that loses mass passes on exit status.
    //
    // the lake-at-rest gate: still water on a bed that is not flat must survive
    // every adapt, because the transfer carries the free surface and not the
    // depth. Reported on either side of each adapt so that a departure made by
    // the time stepping is distinguishable from one made by the adapt.
    char      criterion[32]     = "box";
    PetscBool refine_at_start   = PETSC_FALSE;
    PetscBool allow_coarsening  = PETSC_FALSE;
    PetscBool check_volume      = PETSC_FALSE;
    PetscReal volume_tol        = 1e-11;
    PetscReal box[4]            = {0.0, 1.0, 3.0, 4.0};
    PetscInt  nbox              = 4;
    PetscBool box_set           = PETSC_FALSE;
    PetscReal box_speed         = 0.0;
    PetscReal still_water       = 0.0;
    PetscReal still_water_tol   = 1e-12;
    PetscBool check_still_water = PETSC_FALSE;
    PetscBool check_max_level   = PETSC_FALSE;
    PetscBool report_balance    = PETSC_FALSE;
    // the water a closed domain ends up holding is its initial water plus what
    // its regional sources add, which a case can work out on paper -- so the
    // same figure can be required of every rank count
    PetscReal expected_volume       = 0.0;
    PetscReal expected_volume_tol   = 1e-9;
    PetscBool check_expected_volume = PETSC_FALSE;
    PetscOptionsBegin(comm, NULL, "AMR driver options", "");
    {
      PetscCall(PetscOptionsString("-refine_criterion", "Refinement criterion (box or jump)", "", criterion, criterion, sizeof(criterion), NULL));
      PetscCall(PetscOptionsBool("-refine_at_start", "Adapt to the initial condition before stepping (-refine_criterion jump only)", "",
                                 refine_at_start, &refine_at_start, NULL));
      PetscCall(PetscOptionsBool("-refine_allow_coarsening", "Rebuild the mesh from the base mesh at every adapt to allow coarsening", "",
                                 allow_coarsening, &allow_coarsening, NULL));
      PetscCall(
          PetscOptionsBool("-refine_check_volume", "Check water volume conservation across every adapt", "", check_volume, &check_volume, NULL));
      PetscCall(
          PetscOptionsReal("-refine_check_volume_tol", "Relative tolerance for water volume conservation check", "", volume_tol, &volume_tol, NULL));
      PetscCall(PetscOptionsRealArray("-refine_box", "Box region x0,x1,y0,y1 marked by the box criterion", "", box, &nbox, &box_set));
      PetscCall(PetscOptionsReal("-refine_box_speed", "Speed (m/s) to translate the refinement box along x", "", box_speed, &box_speed, NULL));
      PetscCall(PetscOptionsReal("-refine_still_water", "Start from a lake at rest at elevation eta and check surface departure across adapts", "",
                                 still_water, &still_water, &check_still_water));
      PetscCall(PetscOptionsReal("-refine_still_water_tol", "Relative tolerance for lake-at-rest surface departure check", "", still_water_tol,
                                 &still_water_tol, NULL));
      PetscCall(PetscOptionsBool("-refine_check_max_level", "Report refinement depth after every adapt and fail past -rdy_amr_max_levels + 1", "",
                                 check_max_level, &check_max_level, NULL));
      PetscCall(PetscOptionsBool("-refine_report_balance", "Report max and mean owned cells across ranks before and after every adapt", "",
                                 report_balance, &report_balance, NULL));
      PetscCall(PetscOptionsReal("-refine_expect_water_volume", "Water volume (m^3) the run must end holding", "", expected_volume, &expected_volume,
                                 &check_expected_volume));
      PetscCall(PetscOptionsReal("-refine_expect_water_volume_tol", "Relative tolerance for the final water volume", "", expected_volume_tol,
                                 &expected_volume_tol, NULL));
    }
    PetscOptionsEnd();
    PetscCheck(!box_set || nbox == 4, comm, PETSC_ERR_USER, "-refine_box takes four values x0,x1,y0,y1, got %" PetscInt_FMT, nbox);

    PetscBool by_jump;
    PetscCall(PetscStrcmp(criterion, "jump", &by_jump));
    if (!by_jump) {
      PetscBool by_box;
      PetscCall(PetscStrcmp(criterion, "box", &by_box));
      PetscCheck(by_box, comm, PETSC_ERR_USER, "-refine_criterion must be 'box' or 'jump', got '%s'", criterion);
    }
    if (by_jump) {
      PetscCall(RDyEnableAMRBySurfaceJump(rdy));  // the solution marks the cells
    } else {
      PetscCall(RDyEnableAMR(rdy));
    }

    RefinementDataset refinement_dataset;
    PetscCall(ParseRefinementDataOptions(&refinement_dataset));

    if (check_still_water) {
      PetscCall(SetStillWater(rdy, still_water));
      if (still_water_tol == 0.0) still_water_tol = 1e-12;
    }

    // the refinement-limit gate: no cell may end up more than max_levels
    // bisections below the base cell it descends from
    PetscInt max_levels = 0;
    if (check_max_level) PetscCall(RDyGetAMRMaxLevels(rdy, &max_levels));

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
      if (check_max_level) PetscCall(CheckRefinementLimit(rdy, max_levels));
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
        // nothing to do: RDyPerformAMR computes the indicator and marks the cells
      } else {
        PetscCall(RDyGetTime(rdy, time_unit, &time));
        PetscCall(MarkOwnedCellsForRefinement(rdy, box, box_speed * time));
      }

      PetscReal volume_before = 0.0, volume_after = 0.0;
      if (check_volume) PetscCall(ComputeWaterVolume(rdy, &volume_before));

      PetscReal departure_before = 0.0, departure_after = 0.0;
      if (check_still_water) PetscCall(StillWaterDeparture(rdy, still_water, &departure_before));

      if (report_balance) PetscCall(ReportBalance(rdy, "before"));

      PetscCall(RDyPerformAMR(rdy));

      if (report_balance) PetscCall(ReportBalance(rdy, "after "));

      if (check_still_water) {
        PetscCall(StillWaterDeparture(rdy, still_water, &departure_after));
        if (!myrank) {
          printf("AMR: lake at rest departs by %.3e before the adapt, %.3e after\n", (double)departure_before, (double)departure_after);
        }
        PetscReal tol = still_water_tol * still_water;
        PetscCheck(departure_before <= tol, comm, PETSC_ERR_NOT_CONVERGED,
                   "the time stepping disturbed the lake at rest: %g > %g -- the scheme is not well balanced on this mesh, so the adapt cannot be "
                   "judged",
                   (double)departure_before, (double)tol);
        PetscCheck(departure_after <= tol, comm, PETSC_ERR_NOT_CONVERGED, "the adapt disturbed the lake at rest: %g > %g", (double)departure_after,
                   (double)tol);
      }

      {
        PetscInt ncells;
        PetscCall(RDyGetNumGlobalCells(rdy, &ncells));
        if (!myrank) printf("AMR: %" PetscInt_FMT " cells after adapt\n", ncells);
      }

      if (check_max_level) PetscCall(CheckRefinementLimit(rdy, max_levels));

      if (check_volume) {
        PetscCall(ComputeWaterVolume(rdy, &volume_after));
        PetscReal rel = (volume_before != 0.0) ? PetscAbsReal(volume_after - volume_before) / volume_before : 0.0;
        if (!myrank) printf("AMR: water volume %.15g -> %.15g m^3, relative change %.3e\n", (double)volume_before, (double)volume_after, (double)rel);
        PetscCheck(volume_before > 0.0, comm, PETSC_ERR_NOT_CONVERGED,
                   "the mesh held no water before the adapt, so conservation across it is not being tested");
        PetscCheck(rel <= volume_tol, comm, PETSC_ERR_NOT_CONVERGED, "the adapt did not conserve water: %g > %g relative", (double)rel,
                   (double)volume_tol);
      }

      PetscCall(RDyAdvance(rdy));
    }

    // What the domain should be holding when the run ends, on a closed domain
    // whose only inflow is its regional sources. Asserting a number the case
    // derives on paper, rather than comparing two runs, is what lets the same
    // figure be required of every rank count.
    if (check_expected_volume) {
      PetscReal volume;
      PetscCall(ComputeWaterVolume(rdy, &volume));
      PetscReal rel = PetscAbsReal(volume - expected_volume) / expected_volume;
      if (!myrank) printf("AMR: water volume %.15g m^3 at the end, expected %.15g\n", (double)volume, (double)expected_volume);
      PetscCheck(rel <= expected_volume_tol, comm, PETSC_ERR_NOT_CONVERGED, "the run ended holding %g m^3 against the %g it should: %g > %g relative",
                 (double)volume, (double)expected_volume, (double)rel, (double)expected_volume_tol);
    }

    PetscCall(RDyDestroy(&rdy));
  }

  PetscCall(RDyFinalize());
  return 0;
}
