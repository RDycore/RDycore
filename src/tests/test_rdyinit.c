// This unit test suite tests the initialization and finalization of the
// RDycore library. See https://cmocka.org for details about the CMocka
// testing framework.

#include <rdycore.h>

#include "rdycore_tests.h"

// Globals for capturing command line arguments.
static int    argc_;
static char **argv_;

// Test whether RDyInit works and initializes MPI properly.
static void TestRDyInit(void **state) {
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  assert_false(mpi_initialized);
  assert_int_equal(0, RDyInit(argc_, argv_, "test_rdyinit - RDyInit unit test"));
  assert_true(RDyInitialized() == PETSC_TRUE);
  MPI_Initialized(&mpi_initialized);
  assert_true(mpi_initialized);
}

// Test whether the PETSC_COMM_WORLD communicator behaves properly within cmocka.
static void TestPetscCommWorld(void **state) {
  assert_int_equal(0, RDyInit(argc_, argv_, "test_rdyinit - RDyInit unit test"));

  int num_procs;
  MPI_Comm_size(PETSC_COMM_WORLD, &num_procs);
  assert_true(num_procs >= 1);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  assert_true(rank >= 0);
  assert_true(rank < num_procs);
}

// Test whether MPI performs as expected within CMocka's environment.
static void TestMPIAllreduce(void **state) {
  assert_int_equal(0, RDyInit(argc_, argv_, "test_rdyinit - RDyInit unit test"));

  // Let's see if we can properly sum over all ranks.
  int num_procs;
  MPI_Comm_size(PETSC_COMM_WORLD, &num_procs);

  int one = 1;
  int sum;

  MPI_Allreduce(&one, &sum, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
  assert_int_equal(num_procs, sum);
}

// This function gets printed on finalization.
static void PrintGoodbye(void) { printf("Goodbye!\n"); }

static void TestRDyOnFinalize(void **state) { assert_int_equal(0, RDyOnFinalize(PrintGoodbye)); }

static void TestRDyFinalize(void **state) { assert_int_equal(0, RDyFinalize()); }

int main(int argc, char *argv[]) {
  // Stash command line arguments for usage in tests.
  argc_ = argc;
  argv_ = argv;

  // Define our set of unit tests.
  const struct CMUnitTest tests[] = {
      cmocka_unit_test(TestRDyInit),
      cmocka_unit_test(TestPetscCommWorld),
      cmocka_unit_test(TestMPIAllreduce),
      cmocka_unit_test(TestRDyOnFinalize),
      cmocka_unit_test(TestRDyFinalize),
  };

  // The last two arguments are for setup and teardown functions.
  // (See https://api.cmocka.org/group__cmocka__exec.html)
  return cmocka_run_group_tests(tests, NULL, NULL);
}
