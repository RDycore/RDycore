#include <rdycore.h>

static const char *help_str = "rdycore - a standalone driver for RDycore\n"
"usage: rdycore [options] <filename>\n";

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

  // create rdycore and set it up with the given file
  MPI_Comm comm = PETSC_COMM_WORLD;
  RDy rdy;
  PetscCall(RDyCreate(comm, argv[1], &rdy));
  PetscCall(RDySetup(rdy));

  // Run the simulation to completion.
  PetscCall(RDyRun(rdy));

  // clean up
  PetscCall(RDyDestroy(&rdy));
  PetscCall(RDyFinalize());

  return 0;
}
