#include <private/rdymemoryimpl.h>
#include <petscsys.h>
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

  if (strcmp(argv[1], "-help")) { // if given a config file
    // create rdycore and set it up with the given file
    MPI_Comm comm = PETSC_COMM_WORLD;
    RDy      rdy;
    PetscCall(RDyCreate(comm, argv[1], &rdy));
    PetscCall(RDySetup(rdy));

    // allocate arrays for inspecting simulation data
    PetscInt n;
    PetscCall(RDyGetNumLocalCells(rdy, &n));
    PetscReal *h, *vx, *vy;
    PetscCalloc1(n * sizeof(PetscReal), &h);
    PetscCalloc1(n * sizeof(PetscReal), &vx);
    PetscCalloc1(n * sizeof(PetscReal), &vy);

    // run the simulation to completion using the time parameters in the
    // config file
    PetscReal prev_time, coupling_interval;
    RDyGetTime(rdy, &prev_time);
    RDyGetCouplingInterval(rdy, &coupling_interval);
    while (!RDyFinished(rdy)) { // returns true based on stopping criteria
      // advance the solution by the coupling interval specified in the config file
      PetscCall(RDyAdvance(rdy));

      // the following just check that RDycore is doing the right thing

      PetscReal time, time_step;
      PetscCall(RDyGetTime(rdy, &time));
      PetscCall(RDyGetTimeStep(rdy, &time_step));
      PetscCheck(time > prev_time, comm, PETSC_ERR_USER, "Non-increasing time!");
      PetscCheck(time_step > 0.0, comm, PETSC_ERR_USER, "Non-positive time step!");

      PetscCheck(fabs(time - prev_time + coupling_interval) < 1e-12, comm, PETSC_ERR_USER,
        "RDyAdvance advanced time improperly!");
      prev_time += coupling_interval;

      PetscInt step;
      PetscCall(RDyGetStep(rdy, &step));
      PetscCheck(step > 0, comm, PETSC_ERR_USER, "Non-positive step index!");

      PetscCall(RDyGetHeight(rdy, h));
      PetscCall(RDyGetXVelocity(rdy, vx));
      PetscCall(RDyGetYVelocity(rdy, vy));
    }

    // clean up
    PetscFree(h);
    PetscFree(vx);
    PetscFree(vy);
    PetscCall(RDyDestroy(&rdy));
  }

  PetscCall(RDyFinalize());
  return 0;
}
