// Global assembly test for the analytic SWE RHS Jacobian (increment 2b exit
// test): boots two RDy instances from twin configs -- one with
// numerics.jacobian: fd, one with numerics.jacobian: analytic -- assembles
// both Jacobians at the initial state, and requires the full-matrix relative
// Frobenius difference (boundary rows included) to be FD-limited (< 1e-6).
// Three state/BC combinations: uniform flow (reflecting), dam break
// (reflecting), and dam break with Dirichlet + critical-outflow boundaries.

#include <private/rdycoreimpl.h>
#include <private/rdymeshimpl.h>
#include <rdycore.h>

#include "rdycore_tests.h"

// creates an RDy from the given config, assembles its Jacobian at the IC
// state; the returned RDy owns the returned Mat
static PetscErrorCode AssembleJacobian(const char *yaml, RDy *rdy_out, Mat *J_out) {
  PetscFunctionBeginUser;
  RDy rdy;
  PetscCall(RDyCreate(PETSC_COMM_WORLD, yaml, &rdy));
  PetscCall(RDySetup(rdy));
  PetscCall(TSSetUp(rdy->ts));
  PetscCall(TSComputeRHSJacobian(rdy->ts, 0.0, rdy->u_global, rdy->rhs_jac, rdy->rhs_jac));
  *rdy_out = rdy;
  *J_out   = rdy->rhs_jac;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void CompareConfigs(const char *fd_yaml, const char *an_yaml, const char *label);

static void TestGlobalFDVsAnalytic(void **state) {
  (void)state;
  CompareConfigs("swe_jacobian_global_fd.yaml", "swe_jacobian_global_analytic.yaml", "uniform flow");
}

// dam-break initial state (10 m over 5 m): exercises the exact dissipation
// derivatives on a genuine jump (the frozen approximation failed here)
static void TestGlobalFDVsAnalyticDamBreak(void **state) {
  (void)state;
  CompareConfigs("swe_jacobian_global_dam_fd.yaml", "swe_jacobian_global_dam_analytic.yaml", "dam break");
}

// Dirichlet + critical-outflow boundaries on the dam-break state
static void TestGlobalFDVsAnalyticBCTypes(void **state) {
  (void)state;
  CompareConfigs("swe_jacobian_global_bc_fd.yaml", "swe_jacobian_global_bc_analytic.yaml", "dirichlet+outflow");
}

// ANUGA regularization active (h_anuga = 5 vs h = 10, n = 0.1): the analytic
// Jacobian must carry the regularization terms in both the drag and the
// primitive-reconstruction chain
static void TestGlobalFDVsAnalyticAnugaRegularized(void **state) {
  (void)state;
  CompareConfigs("swe_jacobian_global_anuga_fd.yaml", "swe_jacobian_global_anuga_analytic.yaml", "anuga-regularized drag");
}

// Dirichlet + free-outflow (transmissive) boundaries on the dam-break state:
// the free-outflow ghost copies the interior state, so its ghost map is the
// identity in primitive space
static void TestGlobalFDVsAnalyticFreeOutflow(void **state) {
  (void)state;
  CompareConfigs("swe_jacobian_global_freebc_fd.yaml", "swe_jacobian_global_freebc_analytic.yaml", "dirichlet+free-outflow");
}

static void CompareConfigs(const char *fd_yaml, const char *an_yaml, const char *label) {
  RDy rdy_fd, rdy_an;
  Mat J_fd, J_an;
  assert_int_equal(0, AssembleJacobian(fd_yaml, &rdy_fd, &J_fd));
  assert_int_equal(0, AssembleJacobian(an_yaml, &rdy_an, &J_an));

  // full-matrix comparison: boundary rows included (increment 2d assembles
  // reflecting/Dirichlet/critical-outflow contributions)
  Mat diff;
  assert_int_equal(0, MatDuplicate(J_fd, MAT_COPY_VALUES, &diff));
  assert_int_equal(0, MatAXPY(diff, -1.0, J_an, DIFFERENT_NONZERO_PATTERN));

  PetscReal norm_diff, norm_ref;
  assert_int_equal(0, MatNorm(diff, NORM_FROBENIUS, &norm_diff));
  assert_int_equal(0, MatNorm(J_fd, NORM_FROBENIUS, &norm_ref));
  assert_true(norm_ref > 0.0);

  PetscReal rel_err = norm_diff / norm_ref;
  printf("global FD-vs-analytic full-matrix rel error (%s): %.3e (gate: 1e-6)\n", label, (double)rel_err);
  assert_true(rel_err < 1e-6);

  assert_int_equal(0, MatDestroy(&diff));
  assert_int_equal(0, RDyDestroy(&rdy_fd));
  assert_int_equal(0, RDyDestroy(&rdy_an));
}

static int    argc_ = 0;
static char **argv_ = NULL;

static int Setup(void **state) { return RDyInit(argc_, argv_, "test_swe_jacobian_global - FD vs analytic Jacobian assembly"); }
static int Teardown(void **state) { return RDyFinalize(); }

int main(int argc, char *argv[]) {
  argc_ = argc;
  argv_ = argv;

  const struct CMUnitTest tests[] = {
      cmocka_unit_test(TestGlobalFDVsAnalytic),
      cmocka_unit_test(TestGlobalFDVsAnalyticDamBreak),
      cmocka_unit_test(TestGlobalFDVsAnalyticBCTypes),
      cmocka_unit_test(TestGlobalFDVsAnalyticAnugaRegularized),
      cmocka_unit_test(TestGlobalFDVsAnalyticFreeOutflow),
  };
  return cmocka_run_group_tests(tests, Setup, Teardown);
}
