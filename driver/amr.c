#include <petscsys.h>
#include <rdycore.h>
#include <time.h>

static const char *help_str =
    "rdycore_amr - a standalone RDycore driver for AMR problems\n"
    "usage: rdycore_amr [options] <filename.yaml>\n";

static void usage(const char *exe_name) {
  fprintf(stderr, "%s: usage:\n", exe_name);
  fprintf(stderr, "%s <input.yaml>\n\n", exe_name);
}

typedef struct {
  char dir[PETSC_MAX_PATH_LEN];
  char file[PETSC_MAX_PATH_LEN];

  PetscBool is_defined;             // whether to refine or not
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
    refinement_dataset->date   = (struct tm){
        .tm_year  = date[0] - 1900,
        .tm_mon   = date[1] - 1,
        .tm_mday  = date[2],
        .tm_hour  = date[3],
        .tm_min   = date[4],
        .tm_isdst = -1,
    };

    refinement_dataset->current_date   = (struct tm){
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

static PetscErrorCode MarkLocalCellsForRefinementBasedOnDataset(RDy rdy, Vec global) {

  PetscFunctionBegin;

  PetscInt ncells_local;
  PetscBool *refine_cell;
  PetscReal *area_local;
  PetscScalar *global_ptr;

  // determine the number of local cells
  PetscCall(RDyGetNumLocalCells(rdy, &ncells_local));

  // allocate memory
  PetscCalloc1(ncells_local, &refine_cell);
  PetscCalloc1(ncells_local, &area_local);

  PetscCall(RDyGetLocalCellAreas(rdy, ncells_local, area_local));

  PetscReal area_threshold = 1.0/4.0/2.0; // m^2

  PetscCall(VecGetArray(global, &global_ptr));
  for (PetscInt icell = 0; icell < ncells_local; icell++) {
    if (global_ptr[icell] > 0.0 && area_local[icell] > area_threshold) {
      refine_cell[icell] = PETSC_TRUE;
    } else {
      refine_cell[icell] = PETSC_FALSE;
    }
  }
  PetscCall(VecRestoreArray(global, &global_ptr));

  PetscCall(RDyMarkLocalCellsForRefinement(rdy, ncells_local, refine_cell));

  // free up memory
  PetscFree(refine_cell);
  PetscFree(area_local);

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MarkLocalCellsForRefinement(RDy rdy) {

  PetscFunctionBegin;

  PetscInt ncells_local;
  PetscReal *xc_local, *yc_local;
  PetscBool *refine_cell;

  // determine the number of local cells
  PetscCall(RDyGetNumLocalCells(rdy, &ncells_local));

  // allocate memory
  PetscCalloc1(ncells_local, &xc_local);
  PetscCalloc1(ncells_local, &yc_local);
  PetscCalloc1(ncells_local, &refine_cell);

  // get the x and y coordinates of the local cells
  PetscCall(RDyGetLocalCellXCentroids(rdy, ncells_local, xc_local));
  PetscCall(RDyGetLocalCellYCentroids(rdy, ncells_local, yc_local));

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

  PetscCall(RDyMarkLocalCellsForRefinement(rdy, ncells_local, refine_cell));

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

    RefinementDataset refinement_dataset;
    PetscCall(ParseRefinementDataOptions(&refinement_dataset));

    RDyTimeUnit time_unit;
    PetscReal time, time_step, prev_time;
    PetscCall(RDyGetTime(rdy, time_unit, &prev_time));

    if (refinement_dataset.is_defined) {

      // determine the filename
      PetscCall(DetermineDatasetFilename(&refinement_dataset.current_date, refinement_dataset.dir, refinement_dataset.file));
      printf("filename: %s\n", refinement_dataset.file);
      Vec global;
      PetscCall(RDyReadOneDOFBaseGlobalVecFromBinaryFile(rdy, refinement_dataset.file, &global));

      PetscCall(MarkLocalCellsForRefinementBasedOnDataset(rdy, global));
      PetscCall(VecDestroy(&global));
      PetscCall(RDyRefine(rdy));
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
        printf("filename: %s\n", refinement_dataset.file);
        Vec global_base, global_current;
        PetscCall(RDyReadOneDOFBaseGlobalVecFromBinaryFile(rdy, refinement_dataset.file, &global_base));
        PetscCall(RDyMapOneDOFGlobalBaseVecToCurrentGlobalVec(rdy, global_base, &global_current));
        PetscCall(MarkLocalCellsForRefinementBasedOnDataset(rdy, global_current));

        PetscCall(VecDestroy(&global_base));
        PetscCall(VecDestroy(&global_current));
      }

      PetscCall(RDyRefine(rdy));
      PetscCall(RDyAdvance(rdy));
    }

    PetscCall(RDyDestroy(&rdy));
  }

  PetscCall(RDyFinalize());
  return 0;
}
