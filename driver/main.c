#include <petscsys.h>
#include <rdycore.h>

static const char *help_str =
    "rdycore - a standalone driver for RDycore\n"
    "usage: rdycore [options] <filename>\n"
    "\n"
    "  -report_water_volume   print sum(h * area) over the domain after every\n"
    "                         coupling interval\n";

/// @brief Reports the water the domain holds, sum(h * area) over owned cells.
/// @param [in]  rdy    the simulator
/// @param [in]  n      the number of owned cells
/// @param [in]  h      the owned cells' water depths
/// @param [out] total  the global water volume (m^3), valid on every rank
static PetscErrorCode WaterVolume(RDy rdy, PetscInt n, const PetscReal *h, PetscReal *total) {
  PetscFunctionBeginUser;
  PetscReal *area;
  PetscCall(PetscCalloc1(n, &area));
  PetscCall(RDyGetOwnedCellAreas(rdy, n, area));
  PetscReal local = 0.0;
  for (PetscInt c = 0; c < n; ++c) local += h[c] * area[c];
  PetscCall(PetscFree(area));
  PetscCallMPI(MPI_Allreduce(&local, total, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void usage(const char *exe_name) {
  fprintf(stderr, "%s: usage:\n", exe_name);
  fprintf(stderr, "%s <input.yaml>\n\n", exe_name);
}

int main(int argc, char *argv[]) {
  // print usage info if no arguments given
  if (argc < 2) {
    usage(argv[0]);
    exit(0);
  }

  // initialize subsystems
  PetscCall(RDyInit(argc, argv, help_str));

  // print out our version information
  const char *rdy_build_config;
  PetscCall(RDyGetBuildConfiguration(&rdy_build_config));
  PetscFPrintf(PETSC_COMM_WORLD, stderr, "%s", rdy_build_config);

  if (strcmp(argv[1], "-help")) {  // if given a config file
    // create rdycore and set it up with the given file
    MPI_Comm comm = PETSC_COMM_WORLD;
    RDy      rdy;
    PetscCall(RDyCreate(comm, argv[1], &rdy));

    PetscCall(RDySetup(rdy));

    PetscInt n;
    PetscCall(RDyGetNumOwnedCells(rdy, &n));

    // the water a region's source adds is proportional to that region's area,
    // so this is what tells apart a source applied to the region that declares
    // it from one applied to a region of a different size
    PetscBool report_water_volume = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-report_water_volume", &report_water_volume, NULL));

    // allocate arrays for inspecting simulation data
    PetscReal *h, *hu, *hv;
    PetscCalloc1(n, &h);
    PetscCalloc1(n, &hu);
    PetscCalloc1(n, &hv);

    // run the simulation to completion using the time parameters in the
    // config file
    RDyTimeUnit time_unit;
    PetscCall(RDyGetTimeUnit(rdy, &time_unit));
    PetscReal prev_time, coupling_interval;
    PetscCall(RDyGetTime(rdy, time_unit, &prev_time));
    PetscCall(RDyGetCouplingInterval(rdy, time_unit, &coupling_interval));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-coupling_interval", &coupling_interval, NULL));
    PetscCall(RDySetCouplingInterval(rdy, time_unit, coupling_interval));

    while (!RDyFinished(rdy)) {  // returns true based on stopping criteria

      PetscReal time, time_step;
      PetscCall(RDyGetTime(rdy, time_unit, &time));

      // advance the solution by the coupling interval specified in the config file
      PetscCall(RDyAdvance(rdy));

      // the following just check that RDycore is doing the right thing

      PetscCall(RDyGetTime(rdy, time_unit, &time));
      PetscCall(RDyGetTimeStep(rdy, time_unit, &time_step));
      PetscCheck(time > prev_time, comm, PETSC_ERR_USER, "Non-increasing time!");
      PetscCheck(time_step > 0.0, comm, PETSC_ERR_USER, "Non-positive time step!");

      if (!RDyRestarted(rdy)) {
        PetscCheck(fabs(time - prev_time - coupling_interval) < 1e-12, comm, PETSC_ERR_USER, "RDyAdvance advanced time improperly (%g, %g, %g)!",
                   prev_time, time, fabs(time - prev_time + coupling_interval));
        prev_time += coupling_interval;
      } else {
        prev_time = time;
      }

      PetscInt step;
      PetscCall(RDyGetStep(rdy, &step));
      PetscCheck(step > 0, comm, PETSC_ERR_USER, "Non-positive step index!");

      PetscCall(RDyGetOwnedCellHeights(rdy, n, h));
      PetscCall(RDyGetOwnedCellXMomenta(rdy, n, hu));
      PetscCall(RDyGetOwnedCellYMomenta(rdy, n, hv));

      if (report_water_volume) {
        PetscReal volume;
        PetscCall(WaterVolume(rdy, n, h, &volume));
        PetscCall(PetscPrintf(comm, "water volume %.15g m^3 at t = %g\n", (double)volume, (double)time));
      }
    }

    // clean up
    PetscCall(PetscFree(h));
    PetscCall(PetscFree(hu));
    PetscCall(PetscFree(hv));
    PetscCall(RDyDestroy(&rdy));
  }

  PetscCall(RDyFinalize());
  return 0;
}
