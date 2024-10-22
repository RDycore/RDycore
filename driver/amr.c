#include <petscsys.h>
#include <rdycore.h>

static const char *help_str =
    "rdycore_amr - a standalone RDycore driver for AMR problems\n"
    "usage: rdycore_amr [options] <filename.yaml>\n";

static void usage(const char *exe_name) {
  fprintf(stderr, "%s: usage:\n", exe_name);
  fprintf(stderr, "%s <input.yaml>\n\n", exe_name);
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

    PetscCall(RDyRefine(rdy));

    while (!RDyFinished(rdy)) {
      PetscCall(RDyAdvance(rdy));
    }

    PetscCall(RDyDestroy(&rdy));
  }

  PetscCall(RDyFinalize());
  return 0;
}
