#include <petscsys.h>
#include <rdycore.h>

static const char *help_str =
    "rdycore_mms - a standalone RDycore driver for MMS problems\n"
    "usage: rdycore_mms [options] <filename.yaml>\n";

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

    PetscCall(RDyCreate(comm, argv[1], &rdy));
    PetscCall(RDyMMSSetup(rdy));

    RDyTimeUnit time_unit;
    PetscCall(RDyGetTimeUnit(rdy, &time_unit));
    PetscReal dt, cur_time;
    PetscCall(RDyGetTimeStep(rdy, time_unit, &dt));
    while (!RDyFinished(rdy)) {
      PetscCall(RDyGetTime(rdy, time_unit, &cur_time));

      // enforce dirichlet BCs and compute source terms at half steps
      PetscCall(RDyMMSEnforceBoundaryConditions(rdy, cur_time + 0.5 * dt));
      PetscCall(RDyMMSComputeSourceTerms(rdy, cur_time + 0.5 * dt));

      // advance the solution by the coupling interval specified in the config file
      PetscCall(RDyAdvance(rdy));
    }

    // compute error norms for the final solution
    PetscCall(RDyGetTime(rdy, time_unit, &cur_time));
    PetscReal L1_norms[3], L2_norms[3], Linf_norms[3], global_area;
    PetscInt  num_global_cells;
    PetscCall(RDyMMSComputeErrorNorms(rdy, cur_time, L1_norms, L2_norms, Linf_norms, &num_global_cells, &global_area));

    PetscPrintf(comm, "Avg-cell-area    : %18.16f\n", global_area / num_global_cells);
    PetscPrintf(comm, "Avg-length-scale : %18.16f\n", PetscSqrtReal(global_area / num_global_cells));

    PetscPrintf(comm, "Error-Norm-1     : ");
    for (PetscInt idof = 0; idof < 3; idof++) printf("%18.16f ", L1_norms[idof]);
    PetscPrintf(comm, "\n");

    PetscPrintf(comm, "Error-Norm-2     : ");
    for (PetscInt idof = 0; idof < 3; idof++) printf("%18.16f ", L2_norms[idof]);
    PetscPrintf(comm, "\n");

    PetscPrintf(comm, "Error-Norm-Max   : ");
    for (PetscInt idof = 0; idof < 3; idof++) printf("%18.16f ", Linf_norms[idof]);
    PetscPrintf(comm, "\n");
  }

  PetscCall(RDyFinalize());
  return 0;
}
