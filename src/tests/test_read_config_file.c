// This unit test suite tests the parsing of various configuration files that
// do and don't conform to our input specification
// (https://rdycore.atlassian.net/wiki/spaces/PD/pages/24576001/RDycore+configuration+file)
// See https://cmocka.org for details about the CMocka testing framework.

#include <private/rdycoreimpl.h>
#include <rdycore.h>

#include "rdycore_tests.h"

static int    argc_;
static char **argv_;

// A simple PETSc error handler that prints errors encountered and returns the
// error code received.
static PetscErrorCode PrintErrorHandler(MPI_Comm comm, int line, const char *func, const char *file, PetscErrorCode n, PetscErrorType p,
                                        const char *mess, void *ctx) {
  if (p == PETSC_ERROR_INITIAL) {
    PetscPrintf(comm, "Error encountered in %s (%s, line %d): %s (%d)\n", func, file, line, mess, n);
  }
  return n;
}

// Performs setup
static int Setup(void **state) {
  RDyInit(argc_, argv_, "test_read_config_file - a unit test for RDyCore.");
  PetscPushErrorHandler(PrintErrorHandler, NULL);
  return 0;
}

// Called upon completion of tests
static int Teardown(void **state) { return RDyFinalize(); }

// Here's the thing we're testing.
extern PetscErrorCode ReadConfigFile(RDy rdy);

// Executes ReadConfigFile on the given RDy object using the given string as
// the content of the file. Returns the PETSc error code emitted by the
// function.
static PetscErrorCode ReadConfigString(void **state, RDy rdy, const char *config_string) {
  // write config_string to a file named after the config
  FILE *file;
  assert_int_equal(0, PetscFOpen(rdy->comm, rdy->config_file, "w", &file));
  assert_int_equal(0, PetscFPrintf(rdy->comm, file, "%s\n", config_string));
  assert_int_equal(0, PetscFClose(rdy->comm, file));

  // read the configuration file
  PetscErrorCode ierr = ReadConfigFile(rdy);

  // ReadConfigFile is collective, so reduce the result.
  MPI_Allreduce(MPI_IN_PLACE, &ierr, 1, MPI_INT, MPI_MAX, rdy->comm);

  // remove the file
  /*
  if (rdy->rank == 0) {
    remove(rdy->config_file);
  }*/

  return ierr;
}

static void TestFullSpec(void **state) {
  static const char *config_string =
      "physics:\n"
      "  flow:\n"
      "    mode: swe\n"
      "    bed_friction:\n"
      "      enable: true\n"
      "      model: chezy\n"
      "      coefficient: 1\n"
      "  sediment:\n"
      "    enable: true\n"
      "    d50: 1\n"
      "  salinity:\n"
      "    enable: false\n\n"
      "numerics:\n"
      "  spatial: fv\n"
      "  temporal: euler\n"
      "  riemann: roe\n\n"
      "time:\n"
      "  final_time: 1\n"
      "  unit: years\n"
      "  max_step: 1000\n\n"
      "logging:\n"
      "  file: rdycore.log\n"
      "  level: detail\n\n"
      "restart:\n"
      "  format: h5\n"
      "  frequency: 10\n\n"
      "grid:\n"
      "  file: planar_dam_10x5.msh\n\n"
      "initial_conditions:\n"
      "  1:\n"
      "    flow: dam_top_ic\n"
      "  2:\n"
      "    flow: dam_bottom_ic\n\n"
      "flow_conditions:\n"
      "  dam_top_ic:\n"
      "    type: dirichlet\n"
      "    height: 10\n"
      "    momentum: [0, 0]\n"
      "  dam_bottom_ic:\n"
      "    type: dirichlet\n"
      "    height: 5\n"
      "    momentum: [0, 0]\n";

  RDy rdy;
  assert_int_equal(0, RDyCreate(PETSC_COMM_WORLD, "full_spec", &rdy));
  assert_int_equal(0, ReadConfigString(state, rdy, config_string));

  assert_int_equal(0, RDyDestroy(&rdy));
}

int main(int argc, char *argv[]) {
  // Stash command line arguments for usage in tests.
  argc_ = argc;
  argv_ = argv;

  // Define our set of unit tests.
  const struct CMUnitTest tests[] = {
      cmocka_unit_test(TestFullSpec),
  };

  return cmocka_run_group_tests(tests, Setup, Teardown);
}
