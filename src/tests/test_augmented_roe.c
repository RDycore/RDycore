// This unit test suite tests the pointwise augmented Roe solver used by the
// augmented_roe well-balancing method (src/swe/swe_augmented_roe_flux_*.h).
// See https://cmocka.org for details about the CMocka testing framework.

#include <math.h>
#include <rdycore.h>

#include "../swe/swe_augmented_roe_flux_ceed.h"
#include "../swe/swe_augmented_roe_flux_petsc.h"
#include "../swe/swe_roe_flux_petsc.h"
#include "rdycore_tests.h"

// depth below which a cell is dry (RDycore's default tiny_h)
static const PetscReal TINY_H = 1e-7;

//------------------------------------------------------------------------
// Reference: a verbatim C port of SERGHEI's Solver::roe (src/Solvers.h,
// SERGHEI_POINTWISE_FRICTION != 0, SERGHEI_REBALANCE_SOLVER_CONTRIBUTIONS 0),
// with GRAV = 9.806 and TOLDRY = TINY_H. The dt-dependent wet-wet subcritical
// clamp is compiled out unless SERGHEI_SUBCRITICAL_CLAMP is nonzero, since
// RDycore omits it.
//------------------------------------------------------------------------

#define SERGHEI_SUBCRITICAL_CLAMP 0

static void SergheiRoe(const PetscReal s1[4], const PetscReal s2[4], PetscReal dt, PetscReal dx, PetscReal nx, PetscReal ny, PetscReal upwM[3],
                       PetscReal upwP[3]) {
  const PetscReal GRAV = GRAVITY, SQRTGRAV = sqrt(GRAVITY), TOLDRY = TINY_H, TOL14 = 1e-14;
  PetscReal       lambda[3], lambdaE[3], alpha[3], beta[3], diff[3];
  PetscReal       eigenV[3][3];
  PetscReal       e1, e2, u1, u2, v1, v2;
  PetscReal       un;
  int             k;

  PetscReal betaB = 0.0;
  PetscReal betaF = 0.0;

  PetscReal h1 = s1[0], hu1 = s1[1], hv1 = s1[2], z1 = s1[3];
  PetscReal h2 = s2[0], hu2 = s2[1], hv2 = s2[2], z2 = s2[3];

  PetscReal sqrt1 = sqrt(h1);
  PetscReal sqrt2 = sqrt(h2);

  if (h1 > 0.0) {
    u1 = hu1 / h1;
    v1 = hv1 / h1;
  } else {
    u1 = 0.0;
    v1 = 0.0;
  }
  if (h2 > 0.0) {
    u2 = hu2 / h2;
    v2 = hv2 / h2;
  } else {
    u2 = 0.0;
    v2 = 0.0;
  }

  PetscReal htilde = 0.5 * (h1 + h2);
  PetscReal utilde = (u1 * sqrt1 + u2 * sqrt2) / (sqrt1 + sqrt2);
  PetscReal vtilde = (v1 * sqrt1 + v2 * sqrt2) / (sqrt1 + sqrt2);
  PetscReal ctilde = sqrt(GRAV * htilde);

  un = utilde * nx + vtilde * ny;

  lambda[0] = un - ctilde;
  lambda[1] = un;
  lambda[2] = un + ctilde;

  lambdaE[0] = 0.0;
  lambdaE[1] = 0.0;
  lambdaE[2] = 0.0;

  e1 = u1 * nx + v1 * ny - SQRTGRAV * sqrt1;
  e2 = u2 * nx + v2 * ny - SQRTGRAV * sqrt2;
  if (e1 < 0.0 && e2 > 0.0) {
    lambdaE[0] = lambda[0] - e1 * (e2 - lambda[0]) / (e2 - e1);
    lambda[0]  = e1 * (e2 - lambda[0]) / (e2 - e1);
  }

  e1 = u1 * nx + v1 * ny + SQRTGRAV * sqrt1;
  e2 = u2 * nx + v2 * ny + SQRTGRAV * sqrt2;
  if (e1 < 0.0 && e2 > 0.0) {
    lambdaE[2] = lambda[2] - e2 * (lambda[2] - e1) / (e2 - e1);
    lambda[2]  = e2 * (lambda[2] - e1) / (e2 - e1);
  }

  eigenV[0][0] = 1.0;
  eigenV[0][1] = utilde - ctilde * nx;
  eigenV[0][2] = vtilde - ctilde * ny;
  eigenV[1][0] = 0.0;
  eigenV[1][1] = -ctilde * ny;
  eigenV[1][2] = ctilde * nx;
  eigenV[2][0] = 1.0;
  eigenV[2][1] = utilde + ctilde * nx;
  eigenV[2][2] = vtilde + ctilde * ny;

  diff[0] = h2 - h1;
  diff[1] = hu2 - hu1;
  diff[2] = hv2 - hv1;

  alpha[0] = 0.5 * (diff[0] - ((diff[1] * nx + diff[2] * ny) - un * diff[0]) / ctilde);
  alpha[1] = ((diff[2] - vtilde * diff[0]) * nx - (diff[1] - utilde * diff[0]) * ny) / ctilde;
  alpha[2] = 0.5 * (diff[0] + ((diff[1] * nx + diff[2] * ny) - un * diff[0]) / ctilde);

  PetscReal deltaz = z2 - z1;
  PetscReal l1     = z1 + h1;
  PetscReal l2     = z2 + h2;
  PetscReal dzp    = deltaz;
  PetscReal hp;

  if (deltaz >= 0.0) {
    hp = h1;
    if (l1 < z2) dzp = h1;
  } else {
    hp = h2;
    if (l2 < z1) dzp = -h2;
  }

  betaB = 0.5 / ctilde * GRAV * (hp - 0.5 * fabs(dzp)) * dzp;

  beta[0] = betaB + betaF;

  hp = h1 + alpha[0];
#if SERGHEI_SUBCRITICAL_CLAMP
  if (lambda[0] * lambda[2] < 0.0 && hp > 0.0 && h1 > 0.0 && h2 > 0.0) {
    beta[0] = fmax(beta[0], alpha[0] * lambda[0] - h1 * dx * 0.5 / dt);
    beta[0] = fmin(beta[0], -alpha[2] * lambda[2] + h2 * dx * 0.5 / dt);
  }
#else
  (void)dt;
  (void)dx;
#endif

  beta[1] = 0.0;
  beta[2] = -beta[0];

  for (k = 0; k < 3; k++) {
    upwM[k] = 0.0;
    upwP[k] = 0.0;
  }

  l1 = hp - beta[0] / lambda[0];
  l2 = hp + beta[2] / lambda[2];

  if (l2 < -TOLDRY && h2 < TOLDRY) {
    upwM[0] += (lambda[0] * alpha[0] - beta[0]) + (lambda[2] * alpha[2] - beta[2]);
  } else {
    if (l1 < -TOLDRY && h1 < TOLDRY) {
      upwP[0] += (lambda[0] * alpha[0] - beta[0]) + (lambda[2] * alpha[2] - beta[2]);
    } else {
      for (k = 0; k < 3; k++) {
        if (lambda[k] > 0.0) {
          upwP[0] += (lambda[k] * alpha[k] - beta[k]) * eigenV[k][0];
          upwP[1] += (lambda[k] * alpha[k] - beta[k]) * eigenV[k][1];
          upwP[2] += (lambda[k] * alpha[k] - beta[k]) * eigenV[k][2];
        } else {
          upwM[0] += (lambda[k] * alpha[k] - beta[k]) * eigenV[k][0];
          upwM[1] += (lambda[k] * alpha[k] - beta[k]) * eigenV[k][1];
          upwM[2] += (lambda[k] * alpha[k] - beta[k]) * eigenV[k][2];
        }
      }
    }
  }

  for (k = 0; k < 3; k++) {
    if (lambdaE[k] > 0.0) {
      upwP[0] += (lambdaE[k] * alpha[k]) * eigenV[k][0];
      upwP[1] += (lambdaE[k] * alpha[k]) * eigenV[k][1];
      upwP[2] += (lambdaE[k] * alpha[k]) * eigenV[k][2];
    } else {
      upwM[0] += (lambdaE[k] * alpha[k]) * eigenV[k][0];
      upwM[1] += (lambdaE[k] * alpha[k]) * eigenV[k][1];
      upwM[2] += (lambdaE[k] * alpha[k]) * eigenV[k][2];
    }
  }

  for (k = 0; k < 3; k++) {
    if (fabs(upwM[k]) < TOL14) upwM[k] = 0.0;
    if (fabs(upwP[k]) < TOL14) upwP[k] = 0.0;
  }
}

//------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------

// a conserved state (h, hu, hv) on a bed of elevation z
typedef struct {
  PetscReal h, hu, hv, z;
} State;

// RDycore's (ANUGA-regularized, with h_anuga = 0) velocities
static void Velocities(State s, PetscReal *u, PetscReal *v) {
  *u = (s.h > TINY_H) ? s.hu * s.h / (s.h * s.h) : 0.0;
  *v = (s.h > TINY_H) ? s.hv * s.h / (s.h * s.h) : 0.0;
}

// computes the PETSc-kernel fluxes for the given states and normal
static void PetscFluxes(State L, State R, PetscReal sn, PetscReal cn, PetscReal flux_l[3], PetscReal flux_r[3], PetscReal *amax) {
  PetscReal ul, vl, ur, vr;
  Velocities(L, &ul, &vl);
  Velocities(R, &ur, &vr);
  ComputeSWEAugmentedRoeFluxPointwise(GRAVITY, TINY_H, L.h, ul, vl, L.z, R.h, ur, vr, R.z, sn, cn, flux_l, flux_r, amax);
}

// computes the PETSc-kernel fluctuations for the given states and normal
static void PetscFluctuations(State L, State R, PetscReal sn, PetscReal cn, PetscReal dm[3], PetscReal dp[3]) {
  PetscReal ul, vl, ur, vr, amax;
  Velocities(L, &ul, &vl);
  Velocities(R, &ur, &vr);
  ComputeSWEAugmentedRoeFluctuationsPointwise(GRAVITY, TINY_H, L.h, ul, vl, L.z, R.h, ur, vr, R.z, sn, cn, dm, dp, &amax);
}

// normal flux F_n(U) = (h un, hu un + g h^2 cn / 2, hv un + g h^2 sn / 2)
static void NormalFlux(State s, PetscReal sn, PetscReal cn, PetscReal F[3]) {
  PetscReal u, v;
  Velocities(s, &u, &v);
  PetscReal un = u * cn + v * sn;
  F[0]         = s.h * un;
  F[1]         = s.h * u * un + 0.5 * GRAVITY * s.h * s.h * cn;
  F[2]         = s.h * v * un + 0.5 * GRAVITY * s.h * s.h * sn;
}

// deterministic pseudo-random numbers in [0, 1)
static PetscReal Random(void) {
  static unsigned long long seed = 20261030ULL;
  seed                           = seed * 6364136223846793005ULL + 1442695040888963407ULL;
  return (PetscReal)(seed >> 11) / 9007199254740992.0;
}

// a random state: dry with probability p_dry, otherwise depth in [1e-3, 3],
// velocities in [-umax, umax], bed in [0, 2]
static State RandomState(PetscReal p_dry, PetscReal umax) {
  State s;
  s.z = 2.0 * Random();
  if (Random() < p_dry) {
    s.h = s.hu = s.hv = 0.0;
  } else {
    s.h  = 1e-3 + 3.0 * Random();
    s.hu = s.h * umax * (2.0 * Random() - 1.0);
    s.hv = s.h * umax * (2.0 * Random() - 1.0);
  }
  return s;
}

#define NUM_RANDOM 20000

//------------------------------------------------------------------------
// Tests
//------------------------------------------------------------------------

// 1. wet-wet lake at rest over a step, both orientations and several normals:
//    no fluctuations, and each side sees its own hydrostatic pressure
static void TestLakeAtRestWetWet(void **state) {
  const PetscReal eta = 2.0, angles[] = {0.0, 0.3, M_PI / 2, 2.0, M_PI, -1.0};
  const PetscReal zs[][2] = {
      {0.5, 1.2},
      {1.2, 0.5},
      {1.9, 0.0},
      {0.0, 1.9}
  };
  for (size_t a = 0; a < sizeof(angles) / sizeof(angles[0]); a++) {
    PetscReal sn = sin(angles[a]), cn = cos(angles[a]);
    for (size_t i = 0; i < sizeof(zs) / sizeof(zs[0]); i++) {
      State     L = {eta - zs[i][0], 0.0, 0.0, zs[i][0]};
      State     R = {eta - zs[i][1], 0.0, 0.0, zs[i][1]};
      PetscReal dm[3], dp[3], flux_l[3], flux_r[3], amax;
      PetscFluctuations(L, R, sn, cn, dm, dp);
      for (int j = 0; j < 3; j++) {
        assert_true(fabs(dm[j]) < 1e-13);
        assert_true(fabs(dp[j]) < 1e-13);
      }
      PetscFluxes(L, R, sn, cn, flux_l, flux_r, &amax);
      assert_true(flux_l[0] == 0.0 && flux_r[0] == 0.0);
      assert_true(fabs(flux_l[1] - 0.5 * GRAVITY * L.h * L.h * cn) < 1e-13);
      assert_true(fabs(flux_l[2] - 0.5 * GRAVITY * L.h * L.h * sn) < 1e-13);
      assert_true(fabs(flux_r[1] - 0.5 * GRAVITY * R.h * R.h * cn) < 1e-13);
      assert_true(fabs(flux_r[2] - 0.5 * GRAVITY * R.h * R.h * sn) < 1e-13);
    }
  }
}

// 2. shoreline / dry step at rest: the step acts as a wall (no mass flux, the
//    wet side sees its hydrostatic pressure, the dry side sees nothing)
static void TestShorelineAtRest(void **state) {
  const PetscReal sn = 0.0, cn = 1.0;
  for (int orientation = 0; orientation < 2; orientation++) {
    State     wet = {0.8, 0.0, 0.0, 0.0};  // eta = 0.8
    State     dry = {0.0, 0.0, 0.0, 1.5};  // bed above eta
    State     L = orientation ? dry : wet, R = orientation ? wet : dry;
    PetscReal dm[3], dp[3], flux_l[3], flux_r[3], amax;
    PetscFluctuations(L, R, sn, cn, dm, dp);
    for (int j = 0; j < 3; j++) {
      assert_true(fabs(dm[j]) < 1e-13);
      assert_true(fabs(dp[j]) < 1e-13);
    }
    PetscFluxes(L, R, sn, cn, flux_l, flux_r, &amax);
    assert_true(flux_l[0] == 0.0 && flux_r[0] == 0.0);
    assert_true(fabs(flux_l[1] - 0.5 * GRAVITY * L.h * L.h) < 1e-13);
    assert_true(fabs(flux_r[1] - 0.5 * GRAVITY * R.h * R.h) < 1e-13);
  }
}

// 3. large depth ratio at rest (h_R/h_L = 0.1): well-balanced with the
//    Harten-Hyman entropy fix, while RDycore's critical-flow fix (used by the
//    standard Roe solver) would fire and destroy the balance
static void TestLargeDepthRatioAtRest(void **state) {
  const PetscReal sn = 0.0, cn = 1.0;
  State           L = {1.0, 0.0, 0.0, 0.0};
  State           R = {0.1, 0.0, 0.0, 0.9};
  PetscReal       dm[3], dp[3];
  PetscFluctuations(L, R, sn, cn, dm, dp);
  for (int j = 0; j < 3; j++) {
    assert_true(fabs(dm[j]) < 1e-13);
    assert_true(fabs(dp[j]) < 1e-13);
  }

  // RDycore's critical-flow fix: a1 = |u~n - c~| is replaced when a1 < da1
  PetscReal lambda[3], R_[3][3], dW[3], amax;
  ComputeSWERoeEigenspectrum(L.h, 0.0, 0.0, R.h, 0.0, 0.0, sn, cn, lambda, R_, dW, &amax);
  PetscReal ctilde = sqrt(0.5 * GRAVITY * (L.h + R.h));
  PetscReal da1    = 2.0 * (sqrt(GRAVITY * L.h) - sqrt(GRAVITY * R.h));
  assert_true(ctilde < da1);                              // the fix fires...
  assert_true(fabs(lambda[0] - ctilde) > 1e-3 * ctilde);  // ...and changes |lambda_1|
}

// 4. mass identity: flux_l[0] == flux_r[0] for random states
static void TestMassIdentity(void **state) {
  for (int i = 0; i < NUM_RANDOM; i++) {
    State L = RandomState(0.2, 8.0), R = RandomState(0.2, 8.0);
    if (L.h <= TINY_H && R.h <= TINY_H) continue;
    PetscReal theta = 2.0 * M_PI * Random();
    PetscReal flux_l[3], flux_r[3], amax, dm[3], dp[3], FL[3], FR[3];
    PetscFluxes(L, R, sin(theta), cos(theta), flux_l, flux_r, &amax);
    assert_true(flux_l[0] == flux_r[0]);

    // and the mass fluctuations add up to the jump in normal discharge
    PetscFluctuations(L, R, sin(theta), cos(theta), dm, dp);
    NormalFlux(L, sin(theta), cos(theta), FL);
    NormalFlux(R, sin(theta), cos(theta), FR);
    assert_true(fabs(dm[0] + dp[0] - (FR[0] - FL[0])) < 1e-12 * (1.0 + fabs(FL[0]) + fabs(FR[0])));
  }
}

// 5. flat bed: flux_l == flux_r (the Roe property), and away from transonic
//    states the flux matches the standard Roe solver
static void TestFlatBed(void **state) {
  for (int i = 0; i < NUM_RANDOM; i++) {
    State L = RandomState(0.0, 8.0), R = RandomState(0.0, 8.0);
    R.z             = L.z;
    PetscReal theta = 2.0 * M_PI * Random();
    PetscReal flux_l[3], flux_r[3], amax;
    PetscFluxes(L, R, sin(theta), cos(theta), flux_l, flux_r, &amax);
    PetscReal scale = 1.0 + 0.5 * GRAVITY * (L.h * L.h + R.h * R.h) + fabs(L.hu) + fabs(L.hv) + fabs(R.hu) + fabs(R.hv);
    for (int j = 0; j < 3; j++) assert_true(fabs(flux_l[j] - flux_r[j]) < 1e-12 * scale);
  }

  // subcritical states with small jumps, for which RDycore's critical-flow fix
  // is inactive
  const State states[][2] = {
      {{1.0, 0.3, 0.1, 0.0},  {1.1, 0.275, 0.0, 0.0} },
      {{2.0, -1.0, 0.5, 0.0}, {1.9, -0.8, 0.6, 0.0}  },
      {{0.5, 0.1, -0.2, 0.0}, {0.52, 0.12, -0.2, 0.0}},
  };
  const PetscReal angles[] = {0.0, 0.7, M_PI / 2, 2.5, -2.0};
  for (size_t s = 0; s < sizeof(states) / sizeof(states[0]); s++) {
    for (size_t a = 0; a < sizeof(angles) / sizeof(angles[0]); a++) {
      State     L = states[s][0], R = states[s][1];
      PetscReal sn = sin(angles[a]), cn = cos(angles[a]);
      PetscReal flux_l[3], flux_r[3], amax;
      PetscFluxes(L, R, sn, cn, flux_l, flux_r, &amax);

      RiemannStateData datal, datar;
      PetscReal        ul, vl, ur, vr, roe_flux[3], roe_amax;
      Velocities(L, &ul, &vl);
      Velocities(R, &ur, &vr);
      datal = (RiemannStateData){.num_states = 1, .h = &L.h, .u = &ul, .v = &vl};
      datar = (RiemannStateData){.num_states = 1, .h = &R.h, .u = &ur, .v = &vr};
      assert_int_equal(0, ComputeSWERoeFlux(&datal, &datar, &sn, &cn, roe_flux, &roe_amax));
      for (int j = 0; j < 3; j++) assert_true(fabs(flux_l[j] - roe_flux[j]) < 1e-12);
      assert_true(fabs(amax - roe_amax) < 1e-12);
    }
  }
}

// 6. SERGHEI parity: fluctuations match the reference port for n = (1, 0) and
//    (0, +-1) over random wet/dry, step, and transonic states
static void TestSergheiParity(void **state) {
  const PetscReal normals[][2] = {
      {1.0, 0.0 },
      {0.0, 1.0 },
      {0.0, -1.0},
  };
  int num_transonic = 0, num_wetdry = 0;
  for (int i = 0; i < NUM_RANDOM; i++) {
    State L = RandomState(0.25, 10.0), R = RandomState(0.25, 10.0);
    if (L.h <= TINY_H && R.h <= TINY_H) continue;
    if (L.h <= TINY_H || R.h <= TINY_H) num_wetdry++;
    for (int n = 0; n < 3; n++) {
      PetscReal nx = normals[n][0], ny = normals[n][1];
      PetscReal s1[4] = {L.h, L.hu, L.hv, L.z}, s2[4] = {R.h, R.hu, R.hv, R.z};
      PetscReal upwM[3], upwP[3], dm[3], dp[3];
      SergheiRoe(s1, s2, 1.0, 1.0, nx, ny, upwM, upwP);
      PetscFluctuations(L, R, ny, nx, dm, dp);

      PetscReal ul, vl, ur, vr;
      Velocities(L, &ul, &vl);
      Velocities(R, &ur, &vr);
      PetscReal unl = ul * nx + vl * ny, unr = ur * nx + vr * ny;
      if ((unl - sqrt(GRAVITY * L.h) < 0 && unr - sqrt(GRAVITY * R.h) > 0) || (unl + sqrt(GRAVITY * L.h) < 0 && unr + sqrt(GRAVITY * R.h) > 0)) {
        num_transonic++;
      }

      PetscReal scale = 1.0 + 0.5 * GRAVITY * (L.h * L.h + R.h * R.h) + fabs(L.hu) + fabs(L.hv) + fabs(R.hu) + fabs(R.hv);
      for (int j = 0; j < 3; j++) {
        assert_true(fabs(dm[j] - upwM[j]) < 1e-13 * scale);
        assert_true(fabs(dp[j] - upwP[j]) < 1e-13 * scale);
      }
    }
  }
  // make sure the table exercises the interesting branches
  assert_true(num_transonic > 100);
  assert_true(num_wetdry > 100);
}

// 7. rotation invariance: rotating the normal and the velocities together
//    rotates the fluxes
static void TestRotationInvariance(void **state) {
  for (int i = 0; i < NUM_RANDOM; i++) {
    State L = RandomState(0.2, 8.0), R = RandomState(0.2, 8.0);
    if (L.h <= TINY_H && R.h <= TINY_H) continue;
    PetscReal theta = 2.0 * M_PI * Random(), phi = 2.0 * M_PI * Random();
    PetscReal c = cos(phi), s = sin(phi);

    PetscReal flux_l[3], flux_r[3], amax;
    PetscFluxes(L, R, sin(theta), cos(theta), flux_l, flux_r, &amax);

    State     Lr = {L.h, c * L.hu - s * L.hv, s * L.hu + c * L.hv, L.z};
    State     Rr = {R.h, c * R.hu - s * R.hv, s * R.hu + c * R.hv, R.z};
    PetscReal flux_lr[3], flux_rr[3], amax_r;
    PetscFluxes(Lr, Rr, sin(theta + phi), cos(theta + phi), flux_lr, flux_rr, &amax_r);

    PetscReal scale = 1.0 + 0.5 * GRAVITY * (L.h * L.h + R.h * R.h) + fabs(L.hu) + fabs(L.hv) + fabs(R.hu) + fabs(R.hv);
    assert_true(fabs(flux_lr[0] - flux_l[0]) < 1e-12 * scale);
    assert_true(fabs(flux_lr[1] - (c * flux_l[1] - s * flux_l[2])) < 1e-12 * scale);
    assert_true(fabs(flux_lr[2] - (s * flux_l[1] + c * flux_l[2])) < 1e-12 * scale);
    assert_true(fabs(flux_rr[1] - (c * flux_r[1] - s * flux_r[2])) < 1e-12 * scale);
    assert_true(fabs(flux_rr[2] - (s * flux_r[1] + c * flux_r[2])) < 1e-12 * scale);
    assert_true(fabs(amax_r - amax) < 1e-12 * (1.0 + amax));
  }
}

// 8. PETSc vs CEED kernel drift: identical outputs on the same inputs
static void TestPetscCeedDrift(void **state) {
  for (int i = 0; i < NUM_RANDOM; i++) {
    State L = RandomState(0.2, 10.0), R = RandomState(0.2, 10.0);
    if (L.h <= TINY_H && R.h <= TINY_H) continue;
    PetscReal theta = 2.0 * M_PI * Random();
    PetscReal sn = sin(theta), cn = cos(theta);

    PetscReal flux_l[3], flux_r[3], amax;
    PetscFluxes(L, R, sn, cn, flux_l, flux_r, &amax);

    SWEState   qL = {L.h, L.hu, L.hv}, qR = {R.h, R.hu, R.hv};
    CeedScalar ceed_flux_l[3], ceed_flux_r[3], ceed_amax;
    SWERiemannFlux_AugmentedRoe(GRAVITY, TINY_H, 0.0, qL, qR, L.z, R.z, sn, cn, ceed_flux_l, ceed_flux_r, &ceed_amax);

    for (int j = 0; j < 3; j++) {
      assert_true(flux_l[j] == ceed_flux_l[j]);
      assert_true(flux_r[j] == ceed_flux_r[j]);
    }
    assert_true(amax == ceed_amax);
  }
}

// 9. zero-eigenvalue guard: u~n = +-c~ (lambda_1 = 0 or lambda_3 = 0) with a
//    bed step produces finite fluxes
static void TestZeroEigenvalueGuard(void **state) {
  const PetscReal h = 1.0, c = sqrt(GRAVITY * h);
  for (int sign = -1; sign <= 1; sign += 2) {
    State     L = {h, sign * h * c, 0.0, 0.0};
    State     R = {h, sign * h * c, 0.0, 0.3};
    PetscReal flux_l[3], flux_r[3], amax, dm[3], dp[3];
    PetscFluxes(L, R, 0.0, 1.0, flux_l, flux_r, &amax);
    PetscFluctuations(L, R, 0.0, 1.0, dm, dp);
    for (int j = 0; j < 3; j++) {
      assert_true(isfinite(flux_l[j]) && isfinite(flux_r[j]));
      assert_true(isfinite(dm[j]) && isfinite(dp[j]));
    }
    assert_true(isfinite(amax));
  }
}

static int    argc_ = 0;
static char **argv_ = NULL;

// Called in advance of tests
static int Setup(void **state) { return RDyInit(argc_, argv_, "test_augmented_roe - a unit test for the augmented Roe solver."); }

// Called upon completion of tests
static int Teardown(void **state) { return RDyFinalize(); }

int main(int argc, char *argv[]) {
  // Jot down our command line args to allow PETSc access to them.
  argc_ = argc;
  argv_ = argv;

  // Define our set of unit tests.
  const struct CMUnitTest tests[] = {
      cmocka_unit_test(TestLakeAtRestWetWet),   cmocka_unit_test(TestShorelineAtRest), cmocka_unit_test(TestLargeDepthRatioAtRest),
      cmocka_unit_test(TestMassIdentity),       cmocka_unit_test(TestFlatBed),         cmocka_unit_test(TestSergheiParity),
      cmocka_unit_test(TestRotationInvariance), cmocka_unit_test(TestPetscCeedDrift),  cmocka_unit_test(TestZeroEigenvalueGuard),
  };

  // The last two arguments are for setup and teardown functions.
  // (See https://api.cmocka.org/group__cmocka__exec.html)
  return cmocka_run_group_tests(tests, Setup, Teardown);
}
