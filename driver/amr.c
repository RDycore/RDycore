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

static PetscErrorCode MarkLocalCellsForRefinement(RDy rdy) {

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

    PetscCall(RDyAdvance(rdy));

    while (!RDyFinished(rdy)) {

      PetscCall(MarkLocalCellsForRefinement(rdy));

      PetscCall(RDyRefine(rdy));
      PetscCall(RDyAdvance(rdy));
    }

    PetscCall(RDyDestroy(&rdy));
  }

  PetscCall(RDyFinalize());
  return 0;
}
