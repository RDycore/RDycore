// Edge-level verification harness for the analytic Roe flux Jacobian
// (increment 2a of the Manning-calibration plan): the analytic 3x3 blocks in
// swe_roe_flux_jacobian_petsc.h are compared against central finite
// differences of the actual flux function (swe_roe_flux_petsc.h), with the
// same primitive reconstruction the RHS uses. No mesh, no TS -- states are
// sampled directly.
//
// Tolerances (frozen while increments 2b/2c land -- do not weaken):
//   * FD self-consistency (Richardson): observed order in [1.5, 2.5]
//   * analytic vs FD, near-equal states: rel error < 1e-5 (frozen-dissipation
//     blocks are exact in the limit uR -> uL)
//   * analytic vs FD, distinct states: loose sanity bound (< 0.5 rel) UNTIL
//     increment 2c replaces the frozen blocks with exact ones; then this
//     gate tightens to 1e-6.

#include <private/rdymathimpl.h>
#include <rdycore.h>

#include "../swe/swe_roe_flux_jacobian_petsc.h"
#include "rdycore_tests.h"

static const PetscReal TINY_H  = 1e-7;  // matches the default physics.flow.tiny_h
static const PetscReal H_ANUGA = 0.0;   // matches the default physics.flow.h_anuga_regular

// evaluates the single-edge conservative-in flux map g(consL, consR) -> F[3],
// reconstructing primitives exactly as ComputeRiemannVelocities() does
static PetscErrorCode EdgeFlux(const PetscReal consL[3], const PetscReal consR[3], PetscReal sn, PetscReal cn, PetscReal F[3]) {
  PetscFunctionBeginUser;
  PetscReal qL[3], qR[3], PL[3][3], PR[3][3];
  SWEReconstructPrimitiveWithJacobian(consL, TINY_H, H_ANUGA, qL, PL);
  SWEReconstructPrimitiveWithJacobian(consR, TINY_H, H_ANUGA, qR, PR);

  PetscReal        hl = qL[0], ul = qL[1], vl = qL[2];
  PetscReal        hr = qR[0], ur = qR[1], vr = qR[2];
  RiemannStateData datal = {.num_states = 1, .h = &hl, .u = &ul, .v = &vl};
  RiemannStateData datar = {.num_states = 1, .h = &hr, .u = &ur, .v = &vr};
  PetscReal        amax;
  PetscCall(ComputeSWERoeFlux(&datal, &datar, &sn, &cn, F, &amax));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// central finite difference of the edge flux map: dF/d(consL) and dF/d(consR)
static PetscErrorCode EdgeFluxJacobianFD(const PetscReal consL[3], const PetscReal consR[3], PetscReal sn, PetscReal cn, PetscReal eps_scale,
                                         PetscReal dFdUL[3][3], PetscReal dFdUR[3][3]) {
  PetscFunctionBeginUser;
  for (PetscInt j = 0; j < 3; ++j) {
    PetscReal Lp[3] = {consL[0], consL[1], consL[2]}, Lm[3] = {consL[0], consL[1], consL[2]};
    PetscReal Rp[3] = {consR[0], consR[1], consR[2]}, Rm[3] = {consR[0], consR[1], consR[2]};
    PetscReal eps_l = eps_scale * PetscMax(1.0, PetscAbsReal(consL[j]));
    PetscReal eps_r = eps_scale * PetscMax(1.0, PetscAbsReal(consR[j]));
    Lp[j] += eps_l;
    Lm[j] -= eps_l;
    Rp[j] += eps_r;
    Rm[j] -= eps_r;

    PetscReal Fp[3], Fm[3];
    PetscCall(EdgeFlux(Lp, consR, sn, cn, Fp));
    PetscCall(EdgeFlux(Lm, consR, sn, cn, Fm));
    for (PetscInt i = 0; i < 3; ++i) dFdUL[i][j] = (Fp[i] - Fm[i]) / (2.0 * eps_l);

    PetscCall(EdgeFlux(consL, Rp, sn, cn, Fp));
    PetscCall(EdgeFlux(consL, Rm, sn, cn, Fm));
    for (PetscInt i = 0; i < 3; ++i) dFdUR[i][j] = (Fp[i] - Fm[i]) / (2.0 * eps_r);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// relative Frobenius distance between a pair of block pairs
static PetscReal BlockPairRelError(PetscReal AL[3][3], PetscReal AR[3][3], PetscReal BL[3][3], PetscReal BR[3][3]) {
  PetscReal diff = 0.0, norm = 0.0;
  for (PetscInt i = 0; i < 3; ++i)
    for (PetscInt j = 0; j < 3; ++j) {
      diff += Square(AL[i][j] - BL[i][j]) + Square(AR[i][j] - BR[i][j]);
      norm += Square(BL[i][j]) + Square(BR[i][j]);
    }
  return PetscSqrtReal(diff / PetscMax(norm, 1e-30));
}

// sample states: wet subcritical flow, generic oblique edge normal
static const PetscReal SN = 0.6, CN = 0.8;

// FD harness self-check: Richardson convergence order of the central
// difference on a smooth wet state pair must be ~2
static void TestFDSelfConsistency(void **state) {
  (void)state;
  PetscReal consL[3] = {2.0, 1.2, -0.4}, consR[3] = {1.7, 0.9, 0.3};

  PetscReal J1L[3][3], J1R[3][3], J2L[3][3], J2R[3][3], J4L[3][3], J4R[3][3];
  assert_int_equal(0, EdgeFluxJacobianFD(consL, consR, SN, CN, 1e-4, J1L, J1R));
  assert_int_equal(0, EdgeFluxJacobianFD(consL, consR, SN, CN, 5e-5, J2L, J2R));
  assert_int_equal(0, EdgeFluxJacobianFD(consL, consR, SN, CN, 2.5e-5, J4L, J4R));

  // error(eps) ~ C eps^2: successive differences shrink by ~4
  PetscReal d12 = BlockPairRelError(J1L, J1R, J2L, J2R);
  PetscReal d24 = BlockPairRelError(J2L, J2R, J4L, J4R);
  PetscReal order = PetscLogReal(d12 / d24) / PetscLogReal(2.0);
  assert_true(order > 1.5 && order < 2.5);
}

// frozen-dissipation blocks are exact in the limit uR -> uL: compare against
// FD on a nearly-equal pair
static void TestAnalyticNearEqualStates(void **state) {
  (void)state;
  PetscReal consL[3] = {2.0, 1.2, -0.4};
  PetscReal consR[3] = {2.0 * (1 + 1e-8), 1.2 * (1 - 1e-8), -0.4 * (1 + 1e-8)};

  PetscReal AL[3][3], AR[3][3], FL[3][3], FR[3][3];
  SWERoeFluxJacobian(consL, consR, SN, CN, TINY_H, H_ANUGA, AL, AR);
  assert_int_equal(0, EdgeFluxJacobianFD(consL, consR, SN, CN, 1e-5, FL, FR));

  PetscReal err = BlockPairRelError(AL, AR, FL, FR);
  assert_true(err < 1e-5);
}

// distinct states (wet dam-break-like jump): the exact Jacobian must match
// FD to FD accuracy (increment 2c gate)
static void TestAnalyticDistinctStates(void **state) {
  (void)state;
  PetscReal consL[3] = {10.0, 0.0, 0.0}, consR[3] = {5.0, 0.0, 0.0};

  PetscReal AL[3][3], AR[3][3], FL[3][3], FR[3][3];
  SWERoeFluxJacobian(consL, consR, SN, CN, TINY_H, H_ANUGA, AL, AR);
  assert_int_equal(0, EdgeFluxJacobianFD(consL, consR, SN, CN, 1e-5, FL, FR));

  PetscReal err = BlockPairRelError(AL, AR, FL, FR);
  printf("distinct-state exact-Jacobian rel error: %.3e (gate: 1e-6)\n", (double)err);
  assert_true(err < 1e-6);
}

// states straddling the critical-flow fix and a flowing oblique jump: the
// exact Jacobian must track FD through the entropy-fix branch
static void TestAnalyticTranscriticalStates(void **state) {
  (void)state;
  // right-moving flow near critical speed on the left, subcritical right
  PetscReal consL[3] = {1.0, 3.0, 0.4}, consR[3] = {2.0, 0.5, -0.2};

  PetscReal AL[3][3], AR[3][3], FL[3][3], FR[3][3];
  SWERoeFluxJacobian(consL, consR, SN, CN, TINY_H, H_ANUGA, AL, AR);
  assert_int_equal(0, EdgeFluxJacobianFD(consL, consR, SN, CN, 1e-6, FL, FR));

  PetscReal err = BlockPairRelError(AL, AR, FL, FR);
  printf("transcritical exact-Jacobian rel error: %.3e (gate: 1e-5)\n", (double)err);
  assert_true(err < 1e-5);
}

// both cells dry: flux and Jacobian identically zero
static void TestDryDryStates(void **state) {
  (void)state;
  PetscReal consL[3] = {1e-9, 0.0, 0.0}, consR[3] = {1e-9, 0.0, 0.0};

  PetscReal AL[3][3], AR[3][3];
  SWERoeFluxJacobian(consL, consR, SN, CN, TINY_H, H_ANUGA, AL, AR);
  for (PetscInt i = 0; i < 3; ++i)
    for (PetscInt j = 0; j < 3; ++j) {
      assert_true(AL[i][j] == 0.0);
      assert_true(AR[i][j] == 0.0);
    }
}

// pointwise explicit source map (friction + bed slope momentum rows),
// evaluating the ACTUAL drag implementation shared with ApplySourceExplicit
// and the device source kernel (state-independent external sources omitted --
// they do not contribute to the Jacobian)
static void SourceMap(const PetscReal cons[3], PetscReal n_manning, PetscReal dzdx, PetscReal dzdy, PetscReal h_anuga, PetscReal S[3]) {
  PetscReal h = cons[0], hu = cons[1], hv = cons[2];
  PetscReal bedx = dzdx * 9.806 * h, bedy = dzdy * 9.806 * h;
  PetscReal tbx, tby;
  ComputeSWEManningDrag(h, hu, hv, n_manning, TINY_H, h_anuga, &tbx, &tby);
  S[0] = 0.0;
  S[1] = -bedx - tbx;
  S[2] = -bedy - tby;
}

// analytic source block vs central FD of the pointwise source map
static PetscReal SourceJacobianFDError(const PetscReal cons[3], PetscReal n_manning, PetscReal dzdx, PetscReal dzdy, PetscReal h_anuga,
                                       PetscReal eps_scale) {
  PetscReal D[3][3];
  SWESourceJacobian(cons, n_manning, dzdx, dzdy, TINY_H, h_anuga, D);

  PetscReal diff = 0.0, norm = 0.0;
  PetscReal FD[3][3];
  for (PetscInt j = 0; j < 3; ++j) {
    PetscReal eps  = eps_scale * PetscMax(1.0, PetscAbsReal(cons[j]));
    PetscReal p[3] = {cons[0], cons[1], cons[2]}, m[3] = {cons[0], cons[1], cons[2]};
    p[j] += eps;
    m[j] -= eps;
    PetscReal Sp[3], Sm[3];
    SourceMap(p, n_manning, dzdx, dzdy, h_anuga, Sp);
    SourceMap(m, n_manning, dzdx, dzdy, h_anuga, Sm);
    for (PetscInt i = 0; i < 3; ++i) FD[i][j] = (Sp[i] - Sm[i]) / (2.0 * eps);
  }
  for (PetscInt i = 0; i < 3; ++i)
    for (PetscInt j = 0; j < 3; ++j) {
      diff += Square(D[i][j] - FD[i][j]);
      norm += Square(FD[i][j]);
    }
  return PetscSqrtReal(diff / PetscMax(norm, 1e-30));
}

static void TestSourceJacobian(void **state) {
  (void)state;
  const PetscReal n_manning = 0.03, dzdx = 0.02, dzdy = -0.01;
  PetscReal       cons[3]   = {0.4, 0.25, -0.1};  // shallow, flowing: friction matters

  PetscReal err = SourceJacobianFDError(cons, n_manning, dzdx, dzdy, H_ANUGA, 1e-7);
  assert_true(err < 1e-6);

  // zero-momentum state: friction Jacobian vanishes, bed slope remains
  PetscReal cons0[3] = {0.4, 0.0, 0.0}, D0[3][3];
  SWESourceJacobian(cons0, n_manning, dzdx, dzdy, TINY_H, H_ANUGA, D0);
  assert_true(D0[1][1] == 0.0 && D0[2][2] == 0.0);
  assert_true(PetscAbsReal(D0[1][0] + 9.806 * dzdx) < 1e-12);
}

// ANUGA-regularized drag (h_anuga > 0): the source Jacobian must carry the
// regularization terms exactly, across the depth range where the
// regularization transitions (h >> h_anuga, h ~ h_anuga, h << h_anuga) and
// at NLCD-scale Manning n
static void TestSourceJacobianRegularized(void **state) {
  (void)state;
  const PetscReal h_anuga = 0.005, n_manning = 0.1, dzdx = 0.02, dzdy = -0.01;
  const PetscReal states[][3] = {
      {0.4, 0.25, -0.1},      // h >> h_anuga: near-plain drag
      {0.008, 2e-4, -1e-4},   // h ~ h_anuga: regularization active
      {0.002, 5e-5, 2e-5},    // h < h_anuga: drag heading smoothly to zero
      {2.0e-7, 1e-6, -8e-7},  // just above tiny_h with retained momentum: the old discontinuity site
  };
  for (PetscInt s = 0; s < 4; ++s) {
    PetscReal err = SourceJacobianFDError(states[s], n_manning, dzdx, dzdy, h_anuga, 1e-9);
    assert_true(err < 1e-5);
  }

  // zero-momentum: friction block vanishes, bed slope remains
  PetscReal cons0[3] = {0.01, 0.0, 0.0}, D0[3][3];
  SWESourceJacobian(cons0, n_manning, dzdx, dzdy, TINY_H, h_anuga, D0);
  assert_true(D0[1][1] == 0.0 && D0[2][2] == 0.0);
  assert_true(PetscAbsReal(D0[1][0] + 9.806 * dzdx) < 1e-12);
}

// dS_fric/dn (both plain and regularized) vs central FD in n
static void TestFrictionDN(void **state) {
  (void)state;
  const PetscReal n = 0.1, eps = 1e-7;
  const PetscReal states[][3] = {{0.4, 0.25, -0.1}, {0.008, 2e-4, -1e-4}, {0.002, 5e-5, 2e-5}};
  const PetscReal h_anugas[]  = {0.0, 0.005};
  for (PetscInt a = 0; a < 2; ++a) {
    for (PetscInt s = 0; s < 3; ++s) {
      PetscReal h = states[s][0], hu = states[s][1], hv = states[s][2];
      PetscReal v1, v2, p1, p2, m1;
      SWEFrictionDN(h, hu, hv, n, TINY_H, h_anugas[a], &v1, &v2);
      // FD of S_fric = -tb* in n
      PetscReal tbx_p, tby_p, tbx_m, tby_m;
      ComputeSWEManningDrag(h, hu, hv, n + eps, TINY_H, h_anugas[a], &tbx_p, &tby_p);
      ComputeSWEManningDrag(h, hu, hv, n - eps, TINY_H, h_anugas[a], &tbx_m, &tby_m);
      p1 = -(tbx_p - tbx_m) / (2.0 * eps);
      p2 = -(tby_p - tby_m) / (2.0 * eps);
      m1 = PetscMax(PetscAbsReal(p1), PetscAbsReal(p2));
      assert_true(m1 > 0.0);  // states are wet and moving: derivative must be nonzero
      assert_true(PetscAbsReal(v1 - p1) <= 1e-6 * m1);
      assert_true(PetscAbsReal(v2 - p2) <= 1e-6 * m1);
    }
  }
}

static int    argc_ = 0;
static char **argv_ = NULL;

static int Setup(void **state) { return RDyInit(argc_, argv_, "test_swe_jacobian - edge-level Roe flux Jacobian harness"); }
static int Teardown(void **state) { return RDyFinalize(); }

int main(int argc, char *argv[]) {
  argc_ = argc;
  argv_ = argv;

  const struct CMUnitTest tests[] = {
      cmocka_unit_test(TestFDSelfConsistency),
      cmocka_unit_test(TestAnalyticNearEqualStates),
      cmocka_unit_test(TestAnalyticDistinctStates),
      cmocka_unit_test(TestAnalyticTranscriticalStates),
      cmocka_unit_test(TestDryDryStates),
      cmocka_unit_test(TestSourceJacobian),
      cmocka_unit_test(TestSourceJacobianRegularized),
      cmocka_unit_test(TestFrictionDN),
  };
  return cmocka_run_group_tests(tests, Setup, Teardown);
}
