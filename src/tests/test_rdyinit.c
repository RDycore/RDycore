// This unit test suite tests the initialization and finalization of the
// RDycore library. See https://cmocka.org for details about the CMocka
// testing framework.

#include <rdycore.h>

// CMocka-related includes
#include <cmocka.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

// Globals for capturing command line arguments.
static int    argc_;
static char **argv_;

// Global for indicating whether MPI has already been initialized (we only do
// it once).
static int mpi_initialized_ = 0;

// Test whether RDyInit works and initializes MPI properly.
static void TestRDyInit(void **state) {
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  assert_true(mpi_initialized == mpi_initialized_);
  assert_int_equal(0, RDyInit(argc_, argv_, "test_rdyinit - RDyInit unit test"));
  MPI_Initialized(&mpi_initialized);
  assert_true(mpi_initialized);
  if (mpi_initialized_ != mpi_initialized) {
    mpi_initialized_ = mpi_initialized;
  }

  assert_int_equal(0, RDyFinalize());
}

// Test whether RDyInitNoArguments works and initializes MPI properly.
static void TestRDyInitNoArguments(void **state) {
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  assert_true(mpi_initialized == mpi_initialized_);
  assert_int_equal(0, RDyInitNoArguments());
  MPI_Initialized(&mpi_initialized);
  assert_true(mpi_initialized);
  if (mpi_initialized_ != mpi_initialized) {
    mpi_initialized_ = mpi_initialized;
  }

  assert_int_equal(0, RDyFinalize());
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

  assert_int_equal(0, RDyFinalize());
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

  // This is the last test, so we finalize no matter what.
  assert_int_equal(0, RDyFinalize());
}

int main(int argc, char *argv[]) {
  // Stash command line arguments for usage in tests.
  argc_ = argc;
  argv_ = argv;

  // Define our set of unit tests.
  const struct CMUnitTest tests[] = {
      cmocka_unit_test(TestRDyInit),
      cmocka_unit_test(TestRDyInitNoArguments),
      cmocka_unit_test(TestPetscCommWorld),
      cmocka_unit_test(TestMPIAllreduce),
  };

  // The last two arguments are for setup and teardown functions.
  // (See https://api.cmocka.org/group__cmocka__exec.html)
  return cmocka_run_group_tests(tests, NULL, NULL);
}
