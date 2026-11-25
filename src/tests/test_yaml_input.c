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
  if (rdy->rank == 0) {
    remove(rdy->config_file);
  }

  return ierr;
}

static void TestFullSpec(void **state) {
  static const char *config_string =
      "physics:\n"
      "  flow:\n"
      "    mode: swe\n"
      "  sediment:\n"
      "    num_classes: 1\n"
      "  salinity: false\n\n"
      "numerics:\n"
      "  spatial: fv\n"
      "  temporal: euler\n"
      "  riemann: roe\n\n"
      "time:\n"
      "  stop: 1\n"
      "  unit: years\n"
      "  stop_n: 1000\n\n"
      "logging:\n"
      "  file: rdycore.log\n"
      "  level: detail\n\n"
      "checkpoint:\n"
      "  format: hdf5\n"
      "  interval: 10\n\n"
      "grid:\n"
      "  file: planar_dam_10x5.msh\n\n"
      "regions:\n"
      "  - name: upstream\n"
      "    grid_region_id: 1\n"
      "  - name: downstream\n"
      "    grid_region_id: 2\n\n"
      "surface_composition:\n"
      "  - region: upstream\n"
      "    material: file_material\n"
      "  - region: downstream\n"
      "    material: smooth\n\n"
      "materials:\n"
      "  - name: smooth\n"
      "    properties:\n"
      "      manning:\n"
      "        value: 0.018\n"
      "  - name: file_material\n"
      "    properties:\n"
      "      manning:\n"
      "        file: manning.bin\n"
      "        format: binary\n\n"
      "initial_conditions:\n"
      "  - region: upstream\n"
      "    flow: dam_top_ic\n"
      "    sediment: dam_top_ic\n"
      "  - region: downstream\n"
      "    flow: dam_bottom_ic\n\n"
      "    sediment: dam_bottom_ic\n"
      "boundaries:\n"
      "  - name: top_wall\n"
      "    grid_boundary_id: 1\n\n"
      "boundary_conditions:\n"
      "  - boundaries: [top_wall]\n"
      "    flow: dam_top_ic\n\n"
      "sources:\n"
      "  - region: downstream\n"
      "    flow: dam_bottom_src\n\n"
      "flow_conditions:\n"
      "  - name: dam_top_ic\n"
      "    type: dirichlet\n"
      "    height: 10\n"
      "    x_momentum: 0\n"
      "    y_momentum: 0\n"
      "  - name: dam_bottom_ic\n"
      "    type: dirichlet\n"
      "    height: 5\n"
      "    x_momentum: 0\n\n"
      "    y_momentum: 0\n\n"
      "  - name: dam_bottom_src\n"
      "    type: dirichlet\n"
      "    height: 5\n"
      "    x_momentum: 0\n\n"
      "    y_momentum: 0\n\n"
      "sediment_conditions:\n"
      "  - name: dam_top_ic\n"
      "    type: dirichlet\n"
      "    c0:\n"
      "      value: 0.5\n"
      "  - name: dam_bottom_ic\n"
      "    type: dirichlet\n"
      "    c0:\n"
      "      value: 0.25\n\n"
      "salinity_conditions:\n"
      "  - name: dam_top_ic\n"
      "    type: dirichlet\n"
      "    value: 0.5\n"
      "  - name: dam_bottom_ic\n"
      "    type: dirichlet\n"
      "    value: 0.25\n\n";

  RDy rdy;
  assert_int_equal(0, RDyCreate(PETSC_COMM_WORLD, "full_spec", &rdy));
  assert_int_equal(0, ReadConfigString(state, rdy, config_string));

  assert_int_equal(0, RDyDestroy(&rdy));
}

static void TestBadFlowBCSpec(void **state) {
  static const char *config_string =
      "physics:\n"
      "  flow:\n"
      "    mode: swe\n\n"
      "numerics:\n"
      "  spatial: fv\n"
      "  temporal: euler\n"
      "  riemann: roe\n\n"
      "time:\n"
      "  stop: 1\n"
      "  unit: years\n"
      "  stop_n: 1000\n\n"
      "grid:\n"
      "  file: planar_dam_10x5.msh\n\n"
      "initial_conditions:\n"
      "  1:\n"
      "    flow: dam_top_ic\n"
      "  2:\n"
      "    flow: dam_bottom_ic\n\n"
      "boundary_conditions:\n"
      "  1:\n"
      "    flow: dam_top_ic\n\n"
      "sources:\n"
      "  1:\n"
      "    flow: dam_bottom_src\n\n"
      "flow_conditions:\n"
      "  dam_top_ic:\n"
      "    type: dirichlet\n"
      "  dam_bottom_ic:\n"
      "    height: 5\n";

  RDy rdy;
  assert_int_equal(0, RDyCreate(PETSC_COMM_WORLD, "bad_flow_bc_spec", &rdy));
  assert_int_not_equal(0, ReadConfigString(state, rdy, config_string));  // fail!

  assert_int_equal(0, RDyDestroy(&rdy));
}

// Reads a minimal flow-only config with the given temporal method and optional
// h_anuga_reg_parameter line, returning the resulting h_anuga value.
static PetscReal ReadHAnuga(void **state, const char *temporal, const char *h_anuga_line) {
  char config_string[2048];
  snprintf(config_string, sizeof(config_string),
           "physics:\n"
           "  flow:\n"
           "    mode: swe\n"
           "%s"
           "numerics:\n"
           "  spatial: fv\n"
           "  temporal: %s\n"
           "  riemann: roe\n\n"
           "time:\n"
           "  stop: 1\n"
           "  unit: seconds\n"
           "  stop_n: 10\n\n"
           "grid:\n"
           "  file: planar_dam_10x5.msh\n\n"
           "regions:\n"
           "  - name: domain\n"
           "    grid_region_id: 1\n\n"
           "surface_composition:\n"
           "  - region: domain\n"
           "    material: smooth\n\n"
           "materials:\n"
           "  - name: smooth\n"
           "    properties:\n"
           "      manning:\n"
           "        value: 0.018\n\n"
           "initial_conditions:\n"
           "  - region: domain\n"
           "    flow: still\n\n"
           "flow_conditions:\n"
           "  - name: still\n"
           "    type: dirichlet\n"
           "    height: 1\n"
           "    x_momentum: 0\n"
           "    y_momentum: 0\n",
           h_anuga_line, temporal);
  RDy rdy;
  assert_int_equal(0, RDyCreate(PETSC_COMM_WORLD, "h_anuga_spec", &rdy));
  assert_int_equal(0, ReadConfigString(state, rdy, config_string));
  PetscReal h_anuga = rdy->config.physics.flow.h_anuga_regular;
  assert_int_equal(0, RDyDestroy(&rdy));
  return h_anuga;
}

// Implicit temporal methods default to a non-zero ANUGA desingularization height;
// a value in the YAML file is used as given; explicit methods keep 0 by default.
static void TestImplicitHAnugaDefault(void **state) {
  assert_true(ReadHAnuga(state, "euler", "") == 0.0);
  assert_true(ReadHAnuga(state, "beuler", "") == IMPLICIT_DEFAULT_H_ANUGA);
  assert_true(ReadHAnuga(state, "cn", "") == IMPLICIT_DEFAULT_H_ANUGA);
  assert_true(ReadHAnuga(state, "beuler", "    h_anuga_reg_parameter: 0.05\n") == 0.05);
  assert_true(ReadHAnuga(state, "euler", "    h_anuga_reg_parameter: 0.05\n") == 0.05);
}

int main(int argc, char *argv[]) {
  // Stash command line arguments for usage in tests.
  argc_ = argc;
  argv_ = argv;

  // Define our set of unit tests.
  const struct CMUnitTest tests[] = {
      cmocka_unit_test(TestFullSpec),
      cmocka_unit_test(TestBadFlowBCSpec),
      cmocka_unit_test(TestImplicitHAnugaDefault),
  };

  return cmocka_run_group_tests(tests, Setup, Teardown);
}
