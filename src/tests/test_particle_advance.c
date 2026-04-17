// test_particle_advance.c — Phase 5 integration test for the RDycore particle tracer.
//
// Exercises the full RDyCreate → RDySetup → RDyAdvance loop with
// -particles_per_cell 1 on planar_dam_particles.yaml (3 coupling steps of 0.1 s).
//
// Verifications:
//   1. particles.enabled == PETSC_TRUE after RDySetup
//   2. Global particle count == num_global_cells * particles_per_cell
//   3. All RDyAdvance calls return PETSC_SUCCESS
//   4. After the loop, every local particle coordinate is finite
//
// Run on 1 and 2 MPI ranks.
// Uses the CMocka testing framework (https://cmocka.org).

#include <petscdmswarm.h>
#include <private/rdycoreimpl.h>
#include <private/rdyparticlesimpl.h>
#include <rdycore.h>

#include "rdycore_tests.h"

// Globals for capturing command line arguments.
static int    argc_;
static char **argv_;

// ---------------------------------------------------------------------------
// TestParticleAdvance
//
// Full integration test: RDyCreate → RDySetup → RDyAdvance loop.
// Verifies particle count, advance success, and coordinate finiteness.
// ---------------------------------------------------------------------------
static void TestParticleAdvance(void **state) {
  (void)state;

  PetscErrorCode ierr;

  // ---- 1. Initialize RDycore -----------------------------------------------
  ierr = RDyInit(argc_, argv_, "test_particle_advance");
  assert_int_equal(0, ierr);

  // ---- 2. Create the RDy object from the YAML config ----------------------
  RDy rdy;
  ierr = RDyCreate(PETSC_COMM_WORLD, "planar_dam_particles.yaml", &rdy);
  assert_int_equal(0, ierr);

  // ---- 3. Setup (reads config, builds mesh, creates particle swarm) --------
  ierr = RDySetup(rdy);
  assert_int_equal(0, ierr);

  // ---- 4. Verify particles are enabled ------------------------------------
  assert_true(rdy->particles.enabled == PETSC_TRUE);

  // ---- 5. Verify global particle count == num_global_cells * Npc ----------
  PetscInt num_global_cells;
  ierr = RDyGetNumGlobalCells(rdy, &num_global_cells);
  assert_int_equal(0, ierr);

  PetscInt Npc = rdy->particles.particles_per_cell;
  assert_true(Npc == 1);

  PetscInt global_np;
  ierr = DMSwarmGetSize(rdy->particles.dm_swarm, &global_np);
  assert_int_equal(0, ierr);
  assert_true(global_np == num_global_cells * Npc);

  // ---- 6. Advance loop: run until RDyFinished -----------------------------
  // planar_dam_particles.yaml: stop=0.3 s, coupling_interval=0.1 s → 3 steps
  while (!RDyFinished(rdy)) {
    ierr = RDyAdvance(rdy);
    assert_int_equal(0, ierr);
  }

  // ---- 7. Verify step count -----------------------------------------------
  PetscInt step;
  ierr = RDyGetStep(rdy, &step);
  assert_int_equal(0, ierr);
  assert_true(step == 3);

  // ---- 8. Verify all local particle coordinates are finite ----------------
  DM       sdm = rdy->particles.dm_swarm;
  PetscInt local_np;
  ierr = DMSwarmGetLocalSize(sdm, &local_np);
  assert_int_equal(0, ierr);

  PetscReal *coords;
  ierr = DMSwarmGetField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords);
  assert_int_equal(0, ierr);

  for (PetscInt p = 0; p < local_np; ++p) {
    PetscReal x = coords[p * 2 + 0];
    PetscReal y = coords[p * 2 + 1];
    // Coordinates must be finite (not NaN or Inf)
    assert_true(PetscIsNormalReal(x) || x == 0.0);
    assert_true(PetscIsNormalReal(y) || y == 0.0);
  }

  ierr = DMSwarmRestoreField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords);
  assert_int_equal(0, ierr);

  // ---- 9. Cleanup ---------------------------------------------------------
  ierr = RDyDestroy(&rdy);
  assert_int_equal(0, ierr);

  ierr = RDyFinalize();
  assert_int_equal(0, ierr);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
  // Stash command line arguments for usage in tests.
  argc_ = argc;
  argv_ = argv;

  const struct CMUnitTest tests[] = {
      cmocka_unit_test(TestParticleAdvance),
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
