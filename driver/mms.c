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

    // set things up
    PetscCall(RDyCreate(comm, argv[1], &rdy));
    PetscCall(RDyMMSSetup(rdy));

    // run a convergence study
    PetscInt  num_comps       = 3;
    PetscInt  num_refinements = 3;
    PetscReal L1_conv_rates[num_comps], L2_conv_rates[num_comps], Linf_conv_rates[num_comps];
    PetscCall(RDyMMSEstimateConvergenceRates(rdy, num_refinements, L1_conv_rates, L2_conv_rates, Linf_conv_rates));

    const char *comp_names[3] = {" h", "hu", "hv"};
    PetscPrintf(comm, "Convergence rates:\n");
    for (PetscInt idof = 0; idof < 3; idof++) {
      PetscPrintf(comm, "  %s: L1 = %g, L2 = %g, Linf = %g\n", comp_names[idof], L1_conv_rates[idof], L2_conv_rates[idof], Linf_conv_rates[idof]);
    }

    PetscCall(RDyDestroy(&rdy));

    /*
    // run the problem to completion
    while (!RDyFinished(rdy)) {
      PetscCall(RDyAdvance(rdy));
    }

    // compute error norms for the final solution
    RDyTimeUnit time_unit;
    PetscCall(RDyGetTimeUnit(rdy, &time_unit));
    PetscReal cur_time;
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
    */
  }

  PetscCall(RDyFinalize());
  return 0;
}
